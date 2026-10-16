//! IPL polling phase for MC68000 DIVS.W, recorded against CPU-clock time.
use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus, StepResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Read { address: u32, at: u32, sample: u8 },
    Sync { at: u32, clocks: u32 },
    Poll { at: u32, sample: u8 },
}

struct Bus {
    memory: LinearMemoryBus,
    events: Vec<Event>,
    clock: u32,
    irq_at: Option<u32>,
    held: Option<u8>,
}

impl Bus {
    fn level(&self) -> u8 {
        if self.irq_at.is_some_and(|at| self.clock >= at) {
            3
        } else {
            0
        }
    }
    fn read(&mut self, address: u32) {
        let sample = self.held.unwrap_or_else(|| self.level());
        self.events.push(Event::Read {
            address,
            at: self.clock,
            sample,
        });
        self.clock += 4;
    }
}

impl AddressBus for Bus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory.read_byte(address)
    }
    fn read_word(&mut self, address: u32) -> u16 {
        self.read(address);
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
        self.events.push(Event::Sync {
            at: self.clock,
            clocks,
        });
        self.clock += clocks;
    }
    fn ipl_poll_internal(&mut self) {
        let sample = self.level();
        self.held = Some(sample);
        self.events.push(Event::Poll {
            at: self.clock,
            sample,
        });
    }
    fn ipl_release_sample(&mut self) {
        self.held = None;
    }
}

fn setup(opcode: u16, extension: u16, irq_at: Option<u32>) -> (CpuCore, Bus) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2000);
    cpu.pc = 0x1000;
    cpu.prefetch_queue = [opcode, extension];
    cpu.prefetch_count = 2;
    assert!(cpu.prefetch_enabled());
    let mut memory = LinearMemoryBus::new(0x4000);
    memory.write_word(0x1004, 0x4e71);
    memory.write_word(0x1006, 0x4e71);
    memory.write_word(0x2000, 3);
    (
        cpu,
        Bus {
            memory,
            events: Vec::new(),
            clock: 0,
            irq_at,
            held: None,
        },
    )
}

fn step(cpu: &mut CpuCore, bus: &mut Bus) -> u32 {
    match cpu.step(bus) {
        StepResult::Ok { cycles } => cycles as u32,
        other => panic!("unexpected DIVS result: {other:?}"),
    }
}

fn run(
    opcode: u16,
    extension: u16,
    dividend: i32,
    divisor: i16,
    irq_at: Option<u32>,
) -> (u32, Bus) {
    let (mut cpu, mut bus) = setup(opcode, extension, irq_at);
    cpu.set_d(0, dividend as u32);
    if opcode & 0x003f == 0x003c {
        cpu.prefetch_queue[1] = divisor as u16;
    }
    if opcode & 0x003f == 0x0001 {
        // D1,D0
        cpu.set_d(1, divisor as u16 as u32);
    } else if opcode & 0x003f == 0x0010 {
        // (A0),D0
        cpu.set_a(0, 0x2000);
        bus.memory.write_word(0x2000, divisor as u16);
    }
    let cycles = step(&mut cpu, &mut bus);
    assert_eq!(
        bus.clock, cycles,
        "recorded bus time must equal instruction duration"
    );
    (cycles, bus)
}

fn read_events(bus: &Bus) -> Vec<(u32, u32, u8)> {
    bus.events
        .iter()
        .filter_map(|e| match *e {
            Event::Read {
                address,
                at,
                sample,
            } => Some((address, at, sample)),
            _ => None,
        })
        .collect()
}
fn poll(bus: &Bus) -> Option<(u32, u8)> {
    bus.events.iter().find_map(|e| match *e {
        Event::Poll { at, sample } => Some((at, sample)),
        _ => None,
    })
}
fn last_read_start(bus: &Bus) -> u32 {
    read_events(bus).last().expect("final prefetch").1
}

#[test]
fn divs_nonnegative_dividend_negative_divisor_polls_two_clocks_before_prefetch() {
    // Register, (A0), and immediate source forms. Their operand bus cycle
    // shifts the phase while the handler's internal cycle formula stays fixed.
    for (opcode, extension) in [(0x81c1, 0x4e71), (0x81d0, 0x4e71), (0x81fc, 3)] {
        for (dividend, divisor) in [(100, 3), (100, -3), (0, -3), (-100, 3), (-100, -3)] {
            let (cycles, bus) = run(opcode, extension, dividend, divisor, None);
            let final_fetch = last_read_start(&bus);
            if dividend >= 0 && divisor < 0 {
                let (at, sample) = poll(&bus).unwrap_or_else(|| {
                    panic!(
                        "missing internal poll for {opcode:#06x}, {dividend}/{divisor}: {:?}",
                        bus.events
                    )
                });
                assert_eq!(
                    at + 2,
                    final_fetch,
                    "opcode {opcode:#06x}, {dividend}/{divisor}"
                );
                assert_eq!(sample, 0);
            } else {
                assert_eq!(poll(&bus), None, "ordinary DIVS uses the final bus access");
            }
            assert_eq!(cycles, bus.clock);
            if opcode == 0x81c1 {
                if (dividend, divisor) == (100, -3) {
                    assert_eq!(cycles, 150);
                }
                if (dividend, divisor) == (100, 3) {
                    assert_eq!(cycles, 148);
                }
            }
        }
    }
}

#[test]
fn internal_poll_samples_current_ipl_and_holds_it_over_the_prefetch() {
    // The ordinary final prefetch begins at 146. An IRQ raised at 145 must
    // not be seen by the internal poll at 144; one raised at 144 must be.
    let (_, late) = run(0x81c1, 0x4e71, 100, -3, Some(145));
    assert_eq!(poll(&late), Some((144, 0)));
    assert_eq!(read_events(&late).last().unwrap().2, 0);

    let (_, on_time) = run(0x81c1, 0x4e71, 100, -3, Some(144));
    assert_eq!(poll(&on_time), Some((144, 3)));
    assert_eq!(read_events(&on_time).last().unwrap().2, 3);
}

#[test]
fn divu_keeps_its_ordinary_bus_poll() {
    let (mut cpu, mut bus) = setup(0x80c1, 0x4e71, Some(0)); // DIVU D1,D0
    cpu.set_d(0, 0);
    cpu.set_d(1, 3);
    step(&mut cpu, &mut bus);
    assert_eq!(poll(&bus), None);
    assert_eq!(read_events(&bus).last().unwrap().2, 3);
}

#[test]
fn non_early_absolute_divs_overflow_still_uses_internal_poll() {
    // Quotient overflow does not take the early *absolute* overflow path:
    // high word 1 is less than |divisor| 2.
    let (_, bus) = run(0x81c1, 0x4e71, 100_000, -2, None);
    assert!(poll(&bus).is_some());
}

#[test]
fn early_absolute_overflow_skips_internal_poll() {
    for (dividend, divisor) in [(0x1000_0000, -1), (i32::MIN, -1), (0x1000_0000, 1)] {
        let (_, bus) = run(0x81c1, 0x4e71, dividend, divisor, None);
        assert_eq!(poll(&bus), None);
    }
}
