//! MC68040 floating-point exceptions that the M68040FPSP services: the
//! unimplemented-instruction and unsupported-data-type traps, the state
//! frames FSAVE gives their handlers, and what FRESTORE does with a frame a
//! handler hands back.
//!
//! Active only with `CpuCore::emulate_unimplemented_040` clear on a 68040.
//!
//! The conversion unit (CU) of the 68040 FPU fetches an instruction's
//! operands, converts them to extended precision and tags them. It raises
//! the "E1" exceptions found on operands: unsupported data types here, and
//! SNAN, OPERR and DZ. The normalisation unit (NU) computes and rounds; it
//! raises the "E3" exceptions found on results: OVFL, UNFL and INEX. FSAVE
//! exposes both units' latches as the busy frame; an unimplemented
//! instruction never enters the NU and saves the shorter frame.
//!
//! Sources. The exceptions and their stack frames are M68040UM 9.6.1
//! (unimplemented instructions), 9.6.2 (unsupported data types) and 9.7
//! (state frames). What the frames must hold, and what FRESTORE must do
//! with them, follows the FPSP, which is the only consumer: its get_op
//! (operands, tags, the packed layout), res_func (CU_SAVEPC = $FE restarts
//! the instruction; FSQRT latched as $05), gen_except (E1/E3 frames it
//! restores for the 68040 to report; the shadow FPSR) and the overflow,
//! underflow and operand-error handlers (CMDREG3B, the T flag). The field
//! positions were checked against the displacements a shipped FPSP binary
//! uses, and 68040.library 37.30 runs its handlers on these frames.
//!
//! Timing model: the 68040 holds an exception of an instruction whose
//! destination is a floating-point register pending and takes it when the
//! next floating-point instruction is reached, stacking that instruction's
//! address. This core takes an exception an instruction raises at that
//! instruction's own boundary, stacking the address of the instruction that
//! follows (as it does for the enabled arithmetic exceptions), and keeps the
//! pending form for state loaded by FRESTORE, where the handler that
//! restored it must be able to return first.
//!
//! Not modelled: the 68040 takes overflow and underflow whether or not they
//! are enabled, and operand error and signaling NaN on an FMOVE to an
//! integer, so that the FPSP supplies the result; here those results are
//! computed directly and only enabled exceptions are taken. CU_SAVEPC,
//! CMDREG2B and the register dirty bits of a saved busy frame are zero.

use super::operations::RawSource;
use super::softfloat::ExcFlags;
use super::state_040::{
    E1, E3, Fpu040Frame, Fpu040State, NMC_UNSUP, SFLAG, TFLAG, UFLAG, classify, off, tag,
    tag_unsupported,
};
use crate::core::cpu::CpuCore;
use crate::core::exceptions::vector;
use crate::core::memory::AddressBus;
use crate::fpu::FloatX80;

/// The FP opmodes the 68040 executes in hardware (M68040UM 9.4): FMOVE,
/// FSQRT, FABS, FNEG, FDIV, FADD, FMUL, FSGLDIV, FSGLMUL, FSUB, FCMP, FTST
/// and the single- and double-rounding variants.
fn hardware_opmode(opmode: u16) -> bool {
    matches!(
        opmode,
        0x00 | 0x04
            | 0x18
            | 0x1A
            | 0x20
            | 0x22
            | 0x23
            | 0x24
            | 0x27
            | 0x28
            | 0x38
            | 0x3A
            | 0x40
            | 0x41
            | 0x44
            | 0x45
            | 0x58
            | 0x5A
            | 0x5C
            | 0x5E
            | 0x60
            | 0x62
            | 0x63
            | 0x64
            | 0x66
            | 0x67
            | 0x68
            | 0x6C
    )
}

/// Whether an instruction reads its destination register as an operand.
/// This is the test the FPSP's chk_dy_mo applies to CMDREG1B: opmodes
/// $20-$2F and $60-$6F, and FCMP ($38).
fn dyadic(opmode: u16) -> bool {
    opmode & 0x30 == 0x20 || opmode & 0x7F == 0x38
}

/// A command word as the 68040 latches it: FSQRT ($04) is held as $05.
fn latched_command(w2: u16) -> u16 {
    if w2 & 0xE07F == 0x0004 || w2 & 0xE07F == 0x4004 {
        w2 | 1
    } else {
        w2
    }
}

/// CMDREG3B, the NU's copy of a command word: the register and bits 6, 1
/// and 0 stay, bit 2 moves to bit 5 and bits 5-3 move down one place.
fn cmdreg3b(w2: u16) -> u16 {
    (w2 & 0x03C3) | ((w2 & 0x0038) >> 1) | ((w2 & 0x0004) << 3)
}

/// A source operand as the CU latches it in ETEMP.
#[derive(Clone, Copy)]
struct Latched {
    /// ETEMP: the operand in extended form.
    value: FloatX80,
    tag: u8,
    /// The three longs of a packed decimal source.
    packed: Option<[u32; 3]>,
}

impl Latched {
    fn register(value: FloatX80) -> Self {
        Self {
            value,
            tag: classify(value),
            packed: None,
        }
    }

    /// The image of a fetched source. Integers and normalized singles and
    /// doubles convert exactly. A single or double denormal is tagged as
    /// such and keeps its fraction below a clear integer bit, under the
    /// exponent its zero exponent field converts to ($3F80, $3C00); the
    /// FPSP's get_op replaces that with $3F81 or $3C01 and normalizes.
    fn source(raw: RawSource) -> Self {
        match raw {
            RawSource::Single(bits) if bits & 0x7F80_0000 == 0 && bits & 0x007F_FFFF != 0 => Self {
                value: FloatX80::from_extended(
                    (((bits >> 31) as u16) << 15) | 0x3F80,
                    u64::from(bits & 0x007F_FFFF) << 40,
                ),
                tag: tag::SD_DENORM,
                packed: None,
            },
            RawSource::Double(bits)
                if bits & 0x7FF0_0000_0000_0000 == 0 && bits & 0x000F_FFFF_FFFF_FFFF != 0 =>
            {
                Self {
                    value: FloatX80::from_extended(
                        (((bits >> 63) as u16) << 15) | 0x3C00,
                        (bits & 0x000F_FFFF_FFFF_FFFF) << 11,
                    ),
                    tag: tag::SD_DENORM,
                    packed: None,
                }
            }
            // A NaN is widened as it is: a signaling NaN stays signaling
            // in the frame (the value conversion below would quiet it).
            RawSource::Single(bits)
                if bits & 0x7F80_0000 == 0x7F80_0000 && bits & 0x007F_FFFF != 0 =>
            {
                Self {
                    value: FloatX80::from_extended(
                        (((bits >> 31) as u16) << 15) | 0x7FFF,
                        (1 << 63) | (u64::from(bits & 0x007F_FFFF) << 40),
                    ),
                    tag: tag::NAN,
                    packed: None,
                }
            }
            RawSource::Double(bits)
                if bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000
                    && bits & 0x000F_FFFF_FFFF_FFFF != 0 =>
            {
                Self {
                    value: FloatX80::from_extended(
                        (((bits >> 63) as u16) << 15) | 0x7FFF,
                        (1 << 63) | ((bits & 0x000F_FFFF_FFFF_FFFF) << 11),
                    ),
                    tag: tag::NAN,
                    packed: None,
                }
            }
            RawSource::Packed(longs) => Self {
                value: FloatX80::from_extended(
                    (longs[0] >> 16) as u16,
                    (u64::from(longs[1]) << 32) | u64::from(longs[2]),
                ),
                tag: tag::NORM,
                packed: Some(longs),
            },
            other => Self::register(other.to_x80(&mut ExcFlags::default())),
        }
    }
}

impl CpuCore {
    /// Latch an instruction's operands into the state frame: the command
    /// word, the source in ETEMP with STAG, and the destination register in
    /// FPTEMP with DTAG.
    fn fpu_040_latch(&mut self, frame: Fpu040Frame, w2: u16, src: Latched, dst: FloatX80) {
        let mut st = Fpu040State {
            frame,
            ..Fpu040State::default()
        };
        st.set_word(off::CMDREG1B, latched_command(w2));
        st.set_word(off::CMDREG3B, cmdreg3b(w2));
        st.set_extended(off::ETEMP, src.value);
        st.set_byte(off::STAG, Fpu040State::tag_byte(src.tag, src.value));
        st.set_extended(off::FPTEMP, dst);
        st.set_byte(off::DTAG, Fpu040State::tag_byte(classify(dst), dst));
        if let Some(longs) = src.packed {
            // A packed source is not converted: its first long sits in
            // FPTEMP's last (the FPSP's get_op moves it into place before
            // decoding the twelve bytes at ETEMP).
            st.set_long(off::FPTEMP + 4, longs[1]);
            st.set_long(off::FPTEMP + 8, longs[0]);
        }
        st.set_long(off::FPIARCU, self.ppc);
        self.fpu_040 = st;
    }

    /// Record the state of an unimplemented instruction: the short frame
    /// with the U flag, and E1 when the source is packed decimal.
    fn fpu_040_unimp_state(&mut self, w2: u16, src: Latched, dst: FloatX80) {
        self.fpu_040_latch(Fpu040Frame::Unimp, w2, src, dst);
        self.fpu_040.set_byte(off::T_BYTE, UFLAG);
        if src.packed.is_some() {
            self.fpu_040.set_byte(off::E_BYTE, E1);
        }
    }

    /// Record the state of an unsupported data type: a busy frame with the
    /// CU exception flagged, and the T flag for an FMOVE out.
    fn fpu_040_unsupp_state(&mut self, w2: u16, src: Latched, dst: FloatX80) {
        self.fpu_040_latch(Fpu040Frame::Busy, w2, src, dst);
        self.fpu_040.set_byte(off::E_BYTE, E1);
        self.fpu_040.set_byte(off::NMCEXC, NMC_UNSUP);
        if w2 & 0xE000 == 0x6000 {
            self.fpu_040.set_byte(off::T_BYTE, TFLAG);
        }
    }

    /// FMOVECR on a 68040 with traps: the frame holds only the command.
    pub(super) fn fpu_040_fmovecr_trap<B: AddressBus>(&mut self, bus: &mut B, w2: u16) -> i32 {
        let none = Latched::register(FloatX80::default());
        let dst = self.fpr[((w2 >> 7) & 7) as usize];
        self.fpu_040_unimp_state(w2, none, dst);
        self.take_fp_unimp_060(bus, 0)
    }

    /// The pre-instruction unsupported-data-type exception of an
    /// instruction whose destination is a floating-point register: vector
    /// 55 with the four-word frame. The stacked PC is past the instruction
    /// because the handler finishes it by restoring the repaired frame.
    fn fpu_040_take_unsupp<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.fpiar = self.ppc;
        self.take_fp_arith_exception(bus, vector::FP_UNSUPP_DATA_TYPE, false)
    }

    /// A general instruction with a memory, data-register or immediate
    /// source (the command word already consumed). `None` leaves an opmode
    /// that is neither in hardware nor in the FPSP to the caller.
    pub(super) fn fpu_040_ea_source<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> Option<i32> {
        let src_fmt = (w2 >> 10) & 0x7;
        let dst = ((w2 >> 7) & 7) as usize;
        let opmode = w2 & 0x7F;
        let unimplemented = super::operations::fpu_040_traps_opmode(opmode);
        if !unimplemented && !hardware_opmode(opmode) {
            return None;
        }
        let dst_value = self.fpr[dst];

        if unimplemented {
            // Takes precedence over the operand's data type (M68040UM
            // 9.6.1): the handler finds that in the frame. The operand has
            // been fetched; the format $2 frame carries its address.
            let (src, ea) = match self.fpu_fetch_source(bus, opcode, src_fmt) {
                Some((raw, ea)) => (Latched::source(raw), ea),
                None => (Latched::register(FloatX80::default()), 0),
            };
            self.fpu_040_unimp_state(w2, src, dst_value);
            return Some(self.take_fp_unimp_060(bus, ea));
        }

        let (raw, _) = self.fpu_fetch_source(bus, opcode, src_fmt)?;
        let src = Latched::source(raw);
        if src.packed.is_some()
            || tag_unsupported(src.tag)
            || (dyadic(opmode) && tag_unsupported(classify(dst_value)))
        {
            self.fpu_040_unsupp_state(w2, src, dst_value);
            return Some(self.fpu_040_take_unsupp(bus));
        }

        self.fpiar = self.ppc;
        let mut flags = ExcFlags::default();
        let value = raw.to_x80(&mut flags);
        let cycles = self.fpu_apply_op(opmode, dst, value, flags);
        Some(self.fpu_040_arith_trap(bus, cycles, w2, src.value, dst_value))
    }

    /// A general instruction with a register source (the command word
    /// already consumed and the opmode known to exist).
    pub(super) fn fpu_040_register_source<B: AddressBus>(
        &mut self,
        bus: &mut B,
        w2: u16,
    ) -> Option<i32> {
        let src = Latched::register(self.fpr[((w2 >> 10) & 7) as usize]);
        let dst = ((w2 >> 7) & 7) as usize;
        let opmode = w2 & 0x7F;
        let dst_value = self.fpr[dst];

        if super::operations::fpu_040_traps_opmode(opmode) {
            self.fpu_040_unimp_state(w2, src, dst_value);
            return Some(self.take_fp_unimp_060(bus, 0));
        }
        if !hardware_opmode(opmode) {
            return None;
        }
        if tag_unsupported(src.tag) || (dyadic(opmode) && tag_unsupported(classify(dst_value))) {
            self.fpu_040_unsupp_state(w2, src, dst_value);
            return Some(self.fpu_040_take_unsupp(bus));
        }

        self.fpiar = self.ppc;
        let cycles = self.fpu_apply_op(opmode, dst, src.value, ExcFlags::default());
        Some(self.fpu_040_arith_trap(bus, cycles, w2, src.value, dst_value))
    }

    /// FMOVE FPn,<ea> of a value or to a format the 68040 leaves to
    /// software: packed decimal, or a denormalized or unnormalized source
    /// in any format. A post-instruction exception through vector 55 with
    /// the format $3 frame (M68040UM 9.6.2); nothing is stored. `None` =
    /// the move is done in hardware. The command word is not yet consumed.
    pub(super) fn fpu_040_move_out<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> Option<i32> {
        let dst_fmt = (w2 >> 10) & 0x7;
        let packed = dst_fmt == 3 || dst_fmt == 7;
        let value = self.fpr[((w2 >> 7) & 7) as usize];
        let src = Latched::register(value);
        if !packed && !tag_unsupported(src.tag) {
            return None;
        }
        let _w2 = self.read_imm_16(bus);
        let ea = self.fpu_060_trap_ea(bus, opcode, if packed { 3 } else { dst_fmt });
        // The handler reads the value to move from ETEMP; the exceptional
        // operand of a move out is in FPTEMP as well.
        self.fpu_040_unsupp_state(w2, src, value);
        Some(self.take_fp_unsupp_post_040(bus, ea))
    }

    /// Report an enabled arithmetic exception of the instruction that just
    /// completed, as [`Self::fpu_arith_trap`] does, leaving the busy frame
    /// its handler saves: E1 with the operands for the exceptions found on
    /// them (SNAN, OPERR, DZ), E3 with the result in WBTEMP for those found
    /// on it (OVFL, UNFL, INEX).
    fn fpu_040_arith_trap<B: AddressBus>(
        &mut self,
        bus: &mut B,
        cycles: i32,
        w2: u16,
        src: FloatX80,
        dst_before: FloatX80,
    ) -> i32 {
        if cycles == 0 {
            return 0;
        }
        match self.fpu_enabled_exception(0x7F) {
            Some(vector) => {
                self.fpu_040_arith_state(vector, w2, src, dst_before);
                self.take_fp_arith_exception(bus, vector, false)
            }
            None => cycles,
        }
    }

    /// The busy frame of an arithmetic exception through `vector`.
    fn fpu_040_arith_state(&mut self, vector: u32, w2: u16, src: FloatX80, dst_before: FloatX80) {
        self.fpu_040_latch(Fpu040Frame::Busy, w2, Latched::register(src), dst_before);
        let operand_stage = matches!(
            vector,
            vector::FP_SNAN | vector::FP_OPERR | vector::FP_DZ | vector::FP_BSUN
        );
        if operand_stage {
            self.fpu_040.set_byte(off::E_BYTE, E1);
        } else {
            let result = self.fpr[((w2 >> 7) & 7) as usize];
            self.fpu_040.set_byte(off::E_BYTE, E3);
            self.fpu_040.set_extended(off::WBTEMP, result);
            self.fpu_040
                .set_byte(off::WB_BYTE, u8::from(result.biased_exp() < 0x4000) << 4);
            self.fpu_040.set_long(off::FPSR_SHADOW, self.fpsr);
        }
    }

    /// Before a floating-point instruction that reports exceptions: take
    /// the exception the FPU's state still carries, as a pre-instruction
    /// exception whose stacked PC restarts this instruction. With nothing
    /// to take, the instruction goes ahead and the old state is gone.
    pub(crate) fn fpu_040_pre_instruction<B: AddressBus>(&mut self, bus: &mut B) -> Option<i32> {
        if self.fpu_040.is_idle() {
            return None;
        }
        if self.fpu_040.pending
            && let Some(vector) = self.fpu_040_pending_vector()
        {
            self.fpu_040.pending = false;
            return Some(self.take_exception(bus, vector));
        }
        self.fpu_040.clear();
        None
    }

    /// The vector of the exception a restored frame carries, if any. E1 or
    /// E3 must be set. Enabled exceptions of the FPSR exception byte (the
    /// shadow FPSR's too for E3) are taken by priority; overflow and
    /// underflow are taken whether enabled or not, because the 68040 leaves
    /// their default results to the FPSP; an unsupported data type that is
    /// still flagged goes back through vector 55.
    fn fpu_040_pending_vector(&self) -> Option<u32> {
        let st = &self.fpu_040;
        let e = st.byte(off::E_BYTE);
        if e & (E1 | E3) == 0 {
            return None;
        }
        let busy = st.frame == Fpu040Frame::Busy;
        let mut exc = (self.fpsr >> 8) & 0xFF;
        let mut nmc = 0;
        if busy {
            if e & E3 != 0 {
                exc |= (st.long(off::FPSR_SHADOW) >> 8) & 0xFF;
            }
            nmc = st.byte(off::NMCEXC);
            // NMCEXC bits 1-0 and 6-5: overflow and underflow of either unit.
            exc |= u32::from(((nmc & 0x03) | ((nmc >> 5) & 0x03)) << 3);
        }
        let reportable = (exc & (self.fpcr >> 8) & 0x7F) | (exc & 0x18);
        [
            (0x40, vector::FP_SNAN),
            (0x20, vector::FP_OPERR),
            (0x10, vector::FP_OVFL),
            (0x08, vector::FP_UNFL),
            (0x04, vector::FP_DZ),
            (0x03, vector::FP_INEX),
        ]
        .into_iter()
        .find(|&(bit, _)| reportable & bit != 0)
        .map(|(_, v)| v)
        .or_else(|| {
            (busy && (nmc & NMC_UNSUP != 0 || st.byte(off::NMNEXC) & 0x02 != 0))
                .then_some(vector::FP_UNSUPP_DATA_TYPE)
        })
    }

    /// The longs FSAVE writes for the exceptional state, header first.
    fn fpu_040_frame_longs(&self) -> ([u32; 25], usize) {
        let mut longs = [0u32; 25];
        match self.fpu_040.frame {
            Fpu040Frame::Idle => {
                longs[0] = 0x4100_0000;
                (longs, 1)
            }
            Fpu040Frame::Unimp => {
                longs[0] = 0x4130_0000;
                let first = (off::UNIMP_START - 4) / 4;
                longs[1..13].copy_from_slice(&self.fpu_040.longs[first..]);
                (longs, 13)
            }
            Fpu040Frame::Busy => {
                longs[0] = 0x4160_0000;
                longs[1..25].copy_from_slice(&self.fpu_040.longs);
                (longs, 25)
            }
        }
    }

    /// FSAVE with exceptional state in the FPU: write the unimplemented-
    /// instruction or busy frame (M68040UM 9.7) and leave the FPU idle.
    pub(super) fn fpu_040_fsave<B: AddressBus>(
        &mut self,
        bus: &mut B,
        ea_mode: u8,
        ea_reg: usize,
    ) -> i32 {
        let (longs, count) = self.fpu_040_frame_longs();
        let bytes = 4 * count as u32;
        let addr = match ea_mode {
            3 => {
                let addr = self.a(ea_reg);
                self.set_a(ea_reg, addr.wrapping_add(bytes));
                addr
            }
            4 => {
                let addr = self.a(ea_reg).wrapping_sub(bytes);
                self.set_a(ea_reg, addr);
                addr
            }
            _ => match self.fpu_frame_control_ea(bus, ea_mode, ea_reg, false) {
                Some(addr) => addr,
                None => return 0,
            },
        };
        for (i, long) in longs.iter().take(count).enumerate() {
            self.write_32(bus, addr.wrapping_add(4 * i as u32), *long);
        }
        self.fpu_040.clear();
        8
    }

    /// FRESTORE of a valid non-null frame at `addr`: load the FPU's
    /// exceptional state from it.
    ///
    /// - An idle frame leaves nothing pending.
    /// - A busy frame whose CU_SAVEPC is $FE restarts the instruction in
    ///   CMDREG1B from ETEMP and FPTEMP: this is how the FPSP's
    ///   unsupported-data-type handler has the 68040 finish an instruction
    ///   once it has normalized the operands.
    /// - A frame with E1 or E3 set holds an exception that the next
    ///   floating-point instruction takes.
    /// - A busy frame with neither has nothing left to do; an
    ///   unimplemented-instruction frame is kept as restored.
    pub(super) fn fpu_040_frestore<B: AddressBus>(&mut self, bus: &mut B, addr: u32, header: u32) {
        let size = ((header >> 16) & 0xFF) as usize;
        let mut st = Fpu040State::default();
        match size {
            0x60 => st.frame = Fpu040Frame::Busy,
            0x30 | 0x28 => st.frame = Fpu040Frame::Unimp,
            _ => {
                self.fpu_040.clear();
                return;
            }
        }
        // The frame's last long always lands in the last state long; the
        // version $40 unimplemented frame ($28) lacks the two longs before
        // STAG.
        let first = st.longs.len() - size / 4;
        for i in 0..size / 4 {
            st.longs[first + i] = self.read_32(bus, addr.wrapping_add(4 + 4 * i as u32));
        }
        let e = st.byte(off::E_BYTE);
        if st.frame == Fpu040Frame::Busy && st.byte(off::CU_SAVEPC) == 0xFE {
            self.fpu_040.clear();
            self.fpu_040_resume(&st);
        } else if e & (E1 | E3) != 0 {
            st.pending = true;
            self.fpu_040 = st;
        } else if st.frame == Fpu040Frame::Unimp {
            self.fpu_040 = st;
        } else {
            self.fpu_040.clear();
        }
    }

    /// Run the register-destination instruction a restored busy frame
    /// describes, from the frame's operands. An enabled exception of the
    /// result is held for the next floating-point instruction.
    fn fpu_040_resume(&mut self, st: &Fpu040State) {
        let w2 = st.word(off::CMDREG1B);
        if w2 & 0xA000 != 0 {
            // Only register-destination instructions are restarted.
            return;
        }
        let opmode = match w2 & 0x7F {
            0x05 => 0x04, // FSQRT as the 68040 latches it
            other => other,
        };
        if !hardware_opmode(opmode) {
            return;
        }
        let dst = ((w2 >> 7) & 7) as usize;
        let src = st.operand(off::ETEMP, st.byte(off::STAG) & 0x10 != 0);
        let register = self.fpr[dst];
        if dyadic(opmode) {
            self.fpr[dst] = st.operand(off::FPTEMP, st.byte(off::DTAG) & 0x10 != 0);
        }
        let dst_before = self.fpr[dst];
        // With the S flag the handler passes exception status of its own
        // in the shadow FPSR (the FPSP's INEX1 for a packed source it
        // converted): the instruction reports it along with its own.
        let carried = if st.byte(off::E_BYTE) & SFLAG != 0 {
            ExcFlags((st.long(off::FPSR_SHADOW) >> 8) as u8)
        } else {
            ExcFlags::default()
        };
        self.fpu_apply_op(opmode, dst, src, carried);
        if matches!(opmode, 0x38 | 0x3A) {
            // FCMP and FTST only set the condition codes.
            self.fpr[dst] = register;
        }
        if let Some(vector) = self.fpu_enabled_exception(0x7F) {
            let fpiarcu = st.long(off::FPIARCU);
            self.fpu_040_arith_state(vector, (w2 & !0x7F) | opmode, src, dst_before);
            self.fpu_040.set_long(off::FPIARCU, fpiarcu);
            self.fpu_040.pending = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_words_are_reencoded_for_the_frame() {
        assert_eq!(latched_command(0x0404), 0x0405, "FSQRT FP1,FP0");
        assert_eq!(latched_command(0x4804), 0x4805, "FSQRT.X <ea>,FP0");
        assert_eq!(latched_command(0x0441), 0x0441, "FSSQRT keeps its opmode");
        assert_eq!(latched_command(0x6804), 0x6804, "an FMOVE out k-factor");
        // The FPSP's overflow handler finds FSGLMUL and FSGLDIV as $33 and
        // $30 in CMDREG3B, and the forced precisions as bits 6-5.
        assert_eq!(cmdreg3b(0x0027) & 0x7F, 0x33);
        assert_eq!(cmdreg3b(0x0024) & 0x7F, 0x30);
        assert_eq!(cmdreg3b(0x0062) & 0x60, 0x40);
        assert_eq!(cmdreg3b(0x0066) & 0x60, 0x60);
        assert_eq!(cmdreg3b(0x0380) & 0x0380, 0x0380, "register field kept");
    }

    #[test]
    fn dyadic_matches_the_fpsp_classification() {
        for opmode in [
            0x20, 0x22, 0x23, 0x24, 0x27, 0x28, 0x38, 0x60, 0x67, 0x6C, 0x21,
        ] {
            assert!(dyadic(opmode), "{opmode:#x}");
        }
        for opmode in [0x00, 0x04, 0x18, 0x1A, 0x3A, 0x40, 0x45, 0x5E, 0x0E, 0x30] {
            assert!(!dyadic(opmode), "{opmode:#x}");
        }
    }

    #[test]
    fn frame_operands_take_exponent_bit_15() {
        let mut st = Fpu040State::default();
        // A denormal normalized by one place: exponent -1 in fifteen bits.
        st.set_extended(off::ETEMP, FloatX80::from_extended(0xFFFF, 1 << 63));
        assert_eq!(
            st.operand(off::ETEMP, true),
            FloatX80::from_extended(0x8000, 1 << 62)
        );
        // The same bit over an exponent below $4000 changes nothing.
        st.set_extended(off::ETEMP, FloatX80::from_extended(0x3FFE, 1 << 63));
        assert_eq!(
            st.operand(off::ETEMP, true),
            FloatX80::from_extended(0x3FFE, 1 << 63)
        );
        assert_eq!(
            st.operand(off::ETEMP, false),
            FloatX80::from_extended(0x3FFE, 1 << 63)
        );
        // Shifted out entirely.
        st.set_extended(off::ETEMP, FloatX80::from_extended(0x7F00, 1 << 63));
        assert_eq!(st.operand(off::ETEMP, true), FloatX80::zero(false));
    }

    #[test]
    fn state_bytes_and_words_address_big_endian_longs() {
        let mut st = Fpu040State::default();
        st.set_byte(off::E_BYTE, 0x04);
        st.set_byte(off::T_BYTE, 0x20);
        assert_eq!(st.long(0x48), 0x0420_0000);
        st.set_word(off::CMDREG1B, 0x4822);
        assert_eq!(st.long(0x40), 0x4822_0000);
        assert_eq!(st.word(off::CMDREG1B), 0x4822);
        assert_eq!(st.byte(off::T_BYTE), 0x20);
        st.set_byte(off::NMCEXC, 0x10);
        assert_eq!(st.long(0x38), 0x0010_0000);
    }
}
