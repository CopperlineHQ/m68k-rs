use m68k::core::memory::{BusFault, BusFaultKind};
use m68k::{AddressBus, CpuCore, CpuType, StepResult};

struct PollBus {
    bytes: [u8; 512],
    current_ipl: u8,
    last_sample: u8,
    held_sample: Option<u8>,
    held_operand_sample: Option<u8>,
    operand: (u32, u32),
    writes: usize,
    reads: Vec<(u32, usize)>,
    fault_word: Option<u32>,
    attempted_words: Vec<u32>,
}

impl PollBus {
    fn new(operand: (u32, u32)) -> Self {
        Self {
            bytes: [0; 512],
            current_ipl: 0,
            last_sample: 0,
            held_sample: None,
            held_operand_sample: None,
            operand,
            writes: 0,
            reads: Vec::new(),
            fault_word: None,
            attempted_words: Vec::new(),
        }
    }

    fn sample(&mut self) {
        self.last_sample = self.current_ipl;
    }

    fn read(&mut self, address: u32, len: usize) -> u32 {
        self.sample();
        self.reads.push((address, len));
        let mut value = 0;
        for i in 0..len {
            value = (value << 8) | self.bytes[address as usize + i] as u32;
        }
        if address >= self.operand.0
            && address < self.operand.1
            && address + len as u32 >= self.operand.1
        {
            self.current_ipl = 3;
        }
        value
    }

    fn write(&mut self, address: u32, len: usize, value: u32) {
        self.writes += 1;
        for i in 0..len {
            let shift = (len - i - 1) * 8;
            self.bytes[address as usize + i] = (value >> shift) as u8;
        }
    }
}

impl AddressBus for PollBus {
    fn read_byte(&mut self, a: u32) -> u8 {
        self.read(a, 1) as u8
    }
    fn read_word(&mut self, a: u32) -> u16 {
        self.read(a, 2) as u16
    }
    fn read_long(&mut self, a: u32) -> u32 {
        self.read(a, 4)
    }
    fn write_byte(&mut self, a: u32, v: u8) {
        self.write(a, 1, v as u32)
    }
    fn write_word(&mut self, a: u32, v: u16) {
        self.write(a, 2, v as u32)
    }
    fn write_long(&mut self, a: u32, v: u32) {
        self.write(a, 4, v)
    }

    fn try_read_word(&mut self, address: u32) -> Result<u16, BusFault> {
        self.attempted_words.push(address);
        if self.fault_word == Some(address) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        Ok(self.read_word(address))
    }

    fn ipl_hold_sample(&mut self) {
        self.held_sample = Some(self.last_sample);
    }

    fn ipl_hold_operand_sample(&mut self) {
        self.held_operand_sample = Some(self.last_sample);
    }
}

fn run_tst(
    cpu_type: CpuType,
    size_code: u16,
    ea_bits: u16,
    extension: Option<u16>,
    operand: (u32, u32),
) -> PollBus {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.pc = 0;
    cpu.set_a(0, if ea_bits == 0x0028 { 0x102 } else { 0x100 });
    let mut bus = PollBus::new(operand);
    let opcode = 0x4a00 | size_code | ea_bits;
    bus.bytes[0..2].copy_from_slice(&opcode.to_be_bytes());
    if let Some(ext) = extension {
        bus.bytes[2..4].copy_from_slice(&ext.to_be_bytes());
    }
    bus.bytes[operand.0 as usize..operand.1 as usize].fill(0x12);
    assert!(matches!(cpu.step(&mut bus), StepResult::Ok { .. }));
    bus
}

#[test]
fn memory_tst_holds_operand_sample_for_byte_word_and_long() {
    for (size, len) in [(0x0000, 1), (0x0040, 2), (0x0080, 4)] {
        let bus = run_tst(CpuType::M68000, size, 0x0010, None, (0x100, 0x100 + len));
        assert_eq!(bus.held_operand_sample, Some(0), "size bits {size:#x}");
        assert_eq!(
            bus.held_sample, None,
            "TST uses its distinct operand poll hook"
        );
        assert_eq!(bus.last_sample, 3, "final prefetch should see raised IPL");
        assert_eq!(bus.writes, 0);
        let operand_reads: Vec<_> = bus
            .reads
            .iter()
            .copied()
            .filter(|(address, _)| *address >= 0x100 && *address < 0x104)
            .collect();
        let expected = match len {
            1 => vec![(0x100, 1)],
            2 => vec![(0x100, 2)],
            4 => vec![(0x100, 2), (0x102, 2)],
            _ => unreachable!(),
        };
        assert_eq!(operand_reads, expected, "size bits {size:#x}");
    }
}

#[test]
fn displacement_tst_holds_sample_after_extension_fetch() {
    let bus = run_tst(
        CpuType::M68000,
        0x0040,
        0x0028,
        Some(0xfffe),
        (0x100, 0x102),
    );
    assert_eq!(bus.held_operand_sample, Some(0));
    assert_eq!(bus.held_sample, None);
    assert_eq!(bus.last_sample, 3);
    assert!(
        bus.reads.contains(&(2, 2)),
        "d16 extension must precede the operand read"
    );
    assert!(bus.reads.contains(&(0x100, 2)));
}

#[test]
fn register_and_newer_cpu_tst_keep_boundary_prefetch_sample() {
    let register = run_tst(CpuType::M68000, 0, 0, None, (0x100, 0x101));
    assert_eq!(register.held_sample, None);
    assert_eq!(register.held_operand_sample, None);

    let newer = run_tst(CpuType::M68010, 0x0040, 0x0010, None, (0x100, 0x102));
    assert_eq!(newer.held_sample, None);
    assert_eq!(newer.held_operand_sample, None);
    assert_eq!(newer.last_sample, 3);
}

#[test]
fn tst_long_low_word_fault_does_not_hold_sample_or_update_flags() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.pc = 0;
    cpu.set_a(0, 0x100);
    cpu.set_sp(0x1f0);
    cpu.set_ccr(0x0f);

    let mut bus = PollBus::new((0x100, 0x104));
    bus.bytes[0..2].copy_from_slice(&0x4a90u16.to_be_bytes()); // TST.L (A0)
    bus.bytes[8..12].copy_from_slice(&0x0000_0080u32.to_be_bytes()); // bus-error vector
    bus.bytes[0x100..0x104].copy_from_slice(&0x1234_5678u32.to_be_bytes());
    bus.fault_word = Some(0x102);

    assert!(matches!(cpu.step(&mut bus), StepResult::Ok { .. }));
    assert_eq!(&bus.attempted_words[..3], &[0, 0x100, 0x102]);
    assert_eq!(bus.held_operand_sample, None);
    assert_eq!(
        cpu.get_sr() & 0x0f,
        0x0f,
        "faulting TST must not update CCR"
    );
}
