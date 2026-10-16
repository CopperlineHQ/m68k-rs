//! Signed word division overflow behavior on the MC68000.
use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus, StepResult};

const SR_X: u16 = 0x0010;
const SR_N: u16 = 0x0008;
const SR_Z: u16 = 0x0004;
const SR_V: u16 = 0x0002;
const SR_C: u16 = 0x0001;

fn setup(dividend: u32, divisor: u16, ccr: u16) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x4000);
    bus.write_word(0x1004, 0x4e71);
    bus.write_word(0x1006, 0x4e71);

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2700 | ccr);
    cpu.pc = 0x1000;
    // DIVS.W D1,D0 with a full, steady-state prefetch queue.
    cpu.prefetch_queue = [0x81c1, 0x4e71];
    cpu.prefetch_count = 2;
    cpu.set_d(0, dividend);
    cpu.set_d(1, divisor as u32);
    (cpu, bus)
}

fn step_divs(cpu: &mut CpuCore, bus: &mut LinearMemoryBus) {
    assert!(matches!(cpu.step(bus), StepResult::Ok { .. }));
}

#[test]
fn min_divided_by_minus_one_preserves_register_and_sets_overflow_ccr() {
    for initial_x in [0, SR_X] {
        // V/C are defined and X is preserved. N/Z are undefined by the ISA;
        // check the deterministic overflow values chosen by this core.
        let (mut cpu, mut bus) = setup(0x8000_0000, 0xffff, initial_x | SR_Z | SR_C);
        step_divs(&mut cpu, &mut bus);

        assert_eq!(cpu.d(0), 0x8000_0000, "overflow preserves the dividend");
        let sr = cpu.get_sr();
        assert_ne!(sr & SR_N, 0);
        assert_eq!(sr & SR_Z, 0);
        assert_ne!(sr & SR_V, 0);
        assert_eq!(sr & SR_C, 0);
        assert_eq!(sr & SR_X, initial_x);
    }
}

#[test]
fn negative_boundary_quotient_fits_and_updates_result() {
    let (mut cpu, mut bus) = setup(0xffff_8000, 1, SR_X | SR_V | SR_C);
    step_divs(&mut cpu, &mut bus);

    assert_eq!(cpu.d(0), 0x0000_8000); // remainder 0, quotient -32768
    let sr = cpu.get_sr();
    assert_ne!(sr & SR_N, 0);
    assert_eq!(sr & SR_Z, 0);
    assert_eq!(sr & SR_V, 0);
    assert_eq!(sr & SR_C, 0);
    assert_ne!(sr & SR_X, 0);
}

#[test]
fn other_unrepresentable_quotient_preserves_register_and_sets_v() {
    // -2^31 / -32768 = 65536, outside the signed word quotient range.
    let (mut cpu, mut bus) = setup(0x8000_0000, 0x8000, SR_X);
    step_divs(&mut cpu, &mut bus);

    assert_eq!(cpu.d(0), 0x8000_0000);
    assert_ne!(cpu.get_sr() & SR_V, 0);
    assert_eq!(cpu.get_sr() & SR_X, SR_X);
}
