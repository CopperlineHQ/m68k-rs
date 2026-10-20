//! An access fault inside `run_batch` enters its handler with the same state
//! as under `step` and `execute`.
//!
//! When a data access faults part-way through an instruction, the core takes
//! the exception at once and suppresses the rest of the instruction's bus
//! activity, but the instruction's own register writes still run to the end
//! of its handler: a JSR or RTS still assigns PC, a MOVE still sets the
//! condition codes from the value it never read. The step and execute loops
//! undo that by restoring the handler-entry state the fault dispatch
//! captured; `run_batch` must do the same.

use m68k::core::memory::{AddressBus, BusFault, BusFaultKind};
use m68k::{BatchExit, CpuCore, CpuType, StepResult};

const HANDLER: u32 = 0x3000;
const CODE: u32 = 0x1000;
const TARGET: u32 = 0x2000;
/// Any access at or above this address is a physical bus error.
const FAULT_BASE: u32 = 0x8000;

/// The handler's A-line, distinct from the one at `TARGET`.
const HANDLER_ALINE: u16 = 0xA001;
const TARGET_ALINE: u16 = 0xA002;

struct Bus {
    mem: Vec<u8>,
}

impl Bus {
    fn new() -> Self {
        let mut bus = Self {
            mem: vec![0; FAULT_BASE as usize],
        };
        bus.put_long(8, HANDLER); // vector 2: access fault
        bus.put_words(HANDLER, &[HANDLER_ALINE]);
        bus.put_words(TARGET, &[TARGET_ALINE]);
        bus
    }

    fn put_long(&mut self, at: u32, value: u32) {
        let at = at as usize;
        self.mem[at..at + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn put_words(&mut self, at: u32, words: &[u16]) {
        for (i, w) in words.iter().enumerate() {
            let a = at as usize + 2 * i;
            self.mem[a..a + 2].copy_from_slice(&w.to_be_bytes());
        }
    }

    fn check(address: u32) -> Result<(), BusFault> {
        if address >= FAULT_BASE {
            Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            })
        } else {
            Ok(())
        }
    }
}

impl AddressBus for Bus {
    fn read_byte(&mut self, a: u32) -> u8 {
        self.mem.get(a as usize).copied().unwrap_or(0)
    }
    fn read_word(&mut self, a: u32) -> u16 {
        (u16::from(self.read_byte(a)) << 8) | u16::from(self.read_byte(a + 1))
    }
    fn read_long(&mut self, a: u32) -> u32 {
        (u32::from(self.read_word(a)) << 16) | u32::from(self.read_word(a + 2))
    }
    fn write_byte(&mut self, a: u32, v: u8) {
        if let Some(m) = self.mem.get_mut(a as usize) {
            *m = v;
        }
    }
    fn write_word(&mut self, a: u32, v: u16) {
        self.write_byte(a, (v >> 8) as u8);
        self.write_byte(a + 1, v as u8);
    }
    fn write_long(&mut self, a: u32, v: u32) {
        self.write_word(a, (v >> 16) as u16);
        self.write_word(a + 2, v as u16);
    }
    fn try_read_word(&mut self, a: u32) -> Result<u16, BusFault> {
        Self::check(a)?;
        Ok(self.read_word(a))
    }
    fn try_read_long(&mut self, a: u32) -> Result<u32, BusFault> {
        Self::check(a)?;
        Ok(self.read_long(a))
    }
    fn try_write_long(&mut self, a: u32, v: u32) -> Result<(), BusFault> {
        Self::check(a)?;
        self.write_long(a, v);
        Ok(())
    }
}

fn cpu(program: &[u16], bus: &mut Bus, setup: impl Fn(&mut CpuCore)) -> CpuCore {
    bus.put_words(CODE, program);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68020);
    // Run in user mode, so a faulting user stack still leaves the handler a
    // good supervisor stack for its frame.
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x4000);
    cpu.set_sr(0x0000);
    cpu.set_a(7, 0x6000);
    cpu.pc = CODE;
    setup(&mut cpu);
    cpu
}

/// Run to an A-line with `step`, returning it.
fn stepped(cpu: &mut CpuCore, bus: &mut Bus) -> u16 {
    for _ in 0..100 {
        match cpu.step(bus) {
            StepResult::Ok { .. } => {}
            StepResult::AlineTrap { opcode } => return opcode,
            other => panic!("unexpected {other:?}"),
        }
    }
    panic!("no A-line reached");
}

/// Run to an A-line with `run_batch`, returning it.
fn batched(cpu: &mut CpuCore, bus: &mut Bus) -> u16 {
    match cpu.run_batch(bus, 100, &[]).exit {
        BatchExit::AlineTrap { opcode } => opcode,
        other => panic!("unexpected {other:?}"),
    }
}

fn assert_same_entry(label: &str, a: &CpuCore, b: &CpuCore, bus_a: &Bus, bus_b: &Bus) {
    assert_eq!(a.pc, b.pc, "{label}: pc");
    assert_eq!(a.get_sr(), b.get_sr(), "{label}: sr");
    for i in 0..8 {
        assert_eq!(a.d(i), b.d(i), "{label}: D{i}");
        assert_eq!(a.a(i), b.a(i), "{label}: A{i}");
    }
    assert!(bus_a.mem == bus_b.mem, "{label}: memory (stack frame)");
}

fn compare(label: &str, program: &[u16], setup: impl Fn(&mut CpuCore) + Copy) {
    let mut bus_a = Bus::new();
    let mut a = cpu(program, &mut bus_a, setup);
    let mut bus_b = Bus::new();
    let mut b = cpu(program, &mut bus_b, setup);
    assert_eq!(stepped(&mut a, &mut bus_a), HANDLER_ALINE, "{label}: step");
    assert_eq!(
        batched(&mut b, &mut bus_b),
        HANDLER_ALINE,
        "{label}: run_batch"
    );
    assert_same_entry(label, &a, &b, &bus_a, &bus_b);
}

/// A JSR whose return-address push bus-errors enters the handler; it must
/// not go on to jump to its target.
#[test]
fn jsr_with_a_faulting_push_enters_the_handler() {
    // MOVEA.L #FAULT_BASE+4,A7 (the user stack) ; JSR TARGET.L
    compare(
        "JSR",
        &[0x2E7C, 0x0000, 0x8004, 0x4EB9, 0x0000, TARGET as u16],
        |_| {},
    );
}

/// An RTS whose pop bus-errors enters the handler instead of returning.
#[test]
fn rts_with_a_faulting_pop_enters_the_handler() {
    // MOVEA.L #FAULT_BASE,A7 (the user stack) ; RTS
    compare("RTS", &[0x2E7C, 0x0000, 0x8000, 0x4E75], |_| {});
}

/// A MOVE whose source read bus-errors enters the handler with the
/// condition codes of the faulting instruction's start, not the flags of
/// the zero it never read.
#[test]
fn move_with_a_faulting_read_keeps_the_entry_ccr() {
    // MOVEQ #-1,D0 (N set) ; MOVE.L (A0),D1
    compare("MOVE", &[0x70FF, 0x2210], |cpu| cpu.set_a(0, FAULT_BASE));
}

/// A fault whose exception frame cannot be pushed is a double fault: the
/// CPU halts, and the batch stops there instead of running on.
#[test]
fn double_fault_halts_the_batch() {
    // MOVE.L (A0),D1 from supervisor mode, with the supervisor stack in
    // the faulting region.
    let setup = |cpu: &mut CpuCore| {
        cpu.set_sr(0x2700);
        cpu.set_a(7, FAULT_BASE + 0x100);
        cpu.set_a(0, FAULT_BASE);
    };
    let mut bus_a = Bus::new();
    let mut a = cpu(&[0x2210, 0x4E71], &mut bus_a, setup);
    let mut bus_b = Bus::new();
    let mut b = cpu(&[0x2210, 0x4E71], &mut bus_b, setup);
    assert!(matches!(a.step(&mut bus_a), StepResult::Ok { .. }));
    assert!(a.is_halted(), "step: double fault halts");
    let result = b.run_batch(&mut bus_b, 100, &[]);
    assert_eq!(result.exit, BatchExit::Stopped);
    assert!(b.is_halted(), "run_batch: double fault halts");
    assert_same_entry("double fault", &a, &b, &bus_a, &bus_b);
}
