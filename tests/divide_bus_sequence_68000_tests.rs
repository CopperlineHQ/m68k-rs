//! Bus-phase constraints from OneROM measurements on a PAL MC68000 A500.
use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus, StepResult};

#[derive(Debug, PartialEq)]
enum Event {
    Read(u32),
    Internal(u32),
}

struct Bus {
    memory: LinearMemoryBus,
    events: Vec<Event>,
}

impl AddressBus for Bus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory.read_byte(address)
    }
    fn read_word(&mut self, address: u32) -> u16 {
        self.events.push(Event::Read(address));
        self.memory.read_word(address)
    }
    fn read_long(&mut self, address: u32) -> u32 {
        self.memory.read_long(address)
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory.write_byte(address, value);
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.memory.write_word(address, value);
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.memory.write_long(address, value);
    }
    fn sync(&mut self, clocks: u32) {
        self.events.push(Event::Internal(clocks));
    }
}

fn setup(opcode: u16, next: u16) -> (CpuCore, Bus) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2700);
    cpu.pc = 0x1000;
    // A steady-state queue, avoiding cold-start reads in the measured opcode.
    cpu.prefetch_queue = [opcode, next];
    cpu.prefetch_count = 2;
    assert!(cpu.prefetch_enabled());
    let mut memory = LinearMemoryBus::new(0x4000);
    memory.write_word(0x1004, 0x4e71);
    memory.write_word(0x1006, 0x4e71);
    (
        cpu,
        Bus {
            memory,
            events: Vec::new(),
        },
    )
}

fn step(cpu: &mut CpuCore, bus: &mut Bus) -> u32 {
    match cpu.step(bus) {
        StepResult::Ok { cycles } => cycles as u32,
        other => panic!("unexpected divide result: {other:?}"),
    }
}

#[test]
fn register_divides_compute_before_the_final_prefetch() {
    for opcode in [0x80c1, 0x81c1] {
        // DIVU/DIVS D1,D0
        for dividend in [0, 0x1234_5678, 0xffff_ffff] {
            let (mut cpu, mut bus) = setup(opcode, 0x4e71);
            cpu.set_d(0, dividend);
            cpu.set_d(1, if dividend == 0 { 1 } else { 0x789a });
            let cycles = step(&mut cpu, &mut bus);
            assert_eq!(
                bus.events,
                [Event::Internal(cycles - 4), Event::Read(0x1004)]
            );
        }
    }
}

#[test]
fn memory_operand_precedes_compute_and_final_prefetch() {
    for opcode in [0x80d0, 0x81d0] {
        // DIVU/DIVS (A0),D0
        let (mut cpu, mut bus) = setup(opcode, 0x4e71);
        cpu.set_a(0, 0x2000);
        cpu.set_d(0, 0x1234_5678);
        bus.memory.write_word(0x2000, 0x789a);
        let cycles = step(&mut cpu, &mut bus);
        assert_eq!(
            bus.events,
            [
                Event::Read(0x2000),
                Event::Internal(cycles - 8),
                Event::Read(0x1004)
            ]
        );
    }
}

#[test]
fn immediate_operand_prefetch_precedes_compute() {
    for opcode in [0x80fc, 0x81fc] {
        // DIVU/DIVS #$789a,D0
        let (mut cpu, mut bus) = setup(opcode, 0x789a);
        cpu.set_d(0, 0x1234_5678);
        let cycles = step(&mut cpu, &mut bus);
        assert_eq!(
            bus.events,
            [
                Event::Read(0x1004),
                Event::Internal(cycles - 8),
                Event::Read(0x1006)
            ]
        );
    }
}

#[test]
fn multiplier_keeps_its_early_prefetch() {
    let (mut cpu, mut bus) = setup(0xc0c1, 0x4e71); // MULU D1,D0
    cpu.set_d(0, 0x5678);
    cpu.set_d(1, 0x789a);
    let cycles = step(&mut cpu, &mut bus);
    assert_eq!(
        bus.events,
        [Event::Read(0x1004), Event::Internal(cycles - 4)]
    );
}
