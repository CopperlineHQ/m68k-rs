//! `run_batch` reports cache-control changes to the bus
//! (`AddressBus::cache_control`) before the next fetch, so a host cache
//! model acts on an invalidation where the CPU does, not at the end of the
//! batch.

use m68k::core::memory::AddressBus;
use m68k::{
    BatchExit, CACR_040_DE, CACR_040_IE, CACR_CD, CACR_CI, CACR_ED, CACR_EI, CpuCore, CpuType,
};

const CODE: u32 = 0x1000;
const CODE_END: u32 = 0x1100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Read(u32),
    CacheControl { cacr: u32, caar: u32, strobes: u32 },
}

/// Flat RAM with a host instruction cache over the code range that, like
/// the 68020-68060 caches, does not snoop writes: a word is served from
/// the cache once read, until the bus is told of a CI strobe.
struct CachedCodeBus {
    mem: Vec<u8>,
    lines: Vec<Option<u16>>,
    events: Vec<Event>,
    /// What `cache_control` returns: whether the host acted on the strobes.
    consume: bool,
}

impl CachedCodeBus {
    fn new(words: &[u16], consume: bool) -> Self {
        let mut mem = vec![0; 0x10000];
        for (i, word) in words.iter().enumerate() {
            let at = CODE as usize + i * 2;
            mem[at..at + 2].copy_from_slice(&word.to_be_bytes());
        }
        Self {
            mem,
            lines: vec![None; ((CODE_END - CODE) / 2) as usize],
            events: Vec::new(),
            consume,
        }
    }

    fn memory_word(&self, address: u32) -> u16 {
        let at = (address & 0xFFFF) as usize;
        u16::from_be_bytes([self.mem[at], self.mem[(at + 1) & 0xFFFF]])
    }

    fn cache_control_events(&self) -> Vec<Event> {
        self.events
            .iter()
            .copied()
            .filter(|event| matches!(event, Event::CacheControl { .. }))
            .collect()
    }
}

impl AddressBus for CachedCodeBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        let word = self.read_word(address & !1);
        if address & 1 == 0 {
            (word >> 8) as u8
        } else {
            word as u8
        }
    }
    fn read_word(&mut self, address: u32) -> u16 {
        self.events.push(Event::Read(address));
        if (CODE..CODE_END).contains(&address) && address & 1 == 0 {
            let line = ((address - CODE) / 2) as usize;
            if let Some(word) = self.lines[line] {
                return word;
            }
            let word = self.memory_word(address);
            self.lines[line] = Some(word);
            return word;
        }
        self.memory_word(address)
    }
    fn read_long(&mut self, address: u32) -> u32 {
        ((self.read_word(address) as u32) << 16) | self.read_word(address.wrapping_add(2)) as u32
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        self.mem[(address & 0xFFFF) as usize] = value;
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.write_byte(address, (value >> 8) as u8);
        self.write_byte(address.wrapping_add(1), value as u8);
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.write_word(address, (value >> 16) as u16);
        self.write_word(address.wrapping_add(2), value as u16);
    }
    fn cache_control(&mut self, cacr: u32, caar: u32, strobes: u32) -> bool {
        self.events.push(Event::CacheControl {
            cacr,
            caar,
            strobes,
        });
        if self.consume && strobes & CACR_CI != 0 {
            self.lines.fill(None);
        }
        self.consume
    }
}

fn cpu_at_code(cpu_type: CpuType, cacr: u32) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.pc = CODE;
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x9000);
    cpu.cacr = cacr;
    cpu
}

/// Runs `mulu.l #$33333333,d3` (immediate at $1008), stores $44444444 over
/// the immediate, invalidates with `strobe` (the words placed at $1018)
/// and runs the instruction again. Returns the program and its
/// instruction count.
fn rewrite_and_rerun(strobe: &[u16]) -> (Vec<u16>, u32) {
    let mut words = vec![
        0x7801, // $1000: moveq #1,d4
        0x7601, // $1002: loop: moveq #1,d3
        0x4C3C, 0x3000, 0x3333, 0x3333, // $1004: mulu.l #$33333333,d3
        0x247C, 0x0000, 0x1008, // $100C: movea.l #$1008,a2
        0x24BC, 0x4444, 0x4444, // $1012: move.l #$44444444,(a2)
    ];
    words.extend_from_slice(strobe);
    let disp_at = 0x1018 + 2 * strobe.len() as u32 + 2;
    words.extend_from_slice(&[0x51CC, 0x1002u32.wrapping_sub(disp_at) as u16]); // dbra d4,loop
    words.push(0x60FE); // bra.s *
    let strobe_instructions = if strobe.len() == 1 { 1 } else { 2 };
    (words, 1 + 2 * (5 + strobe_instructions))
}

/// move.l #(ED|CI|EI),d0 ; movec d0,cacr
const MOVEC_CLEAR_030: [u16; 5] = [0x203C, 0x0000, 0x0109, 0x4E7B, 0x0002];

#[test]
fn movec_cacr_clear_reaches_the_bus_before_the_next_fetch() {
    let (words, count) = rewrite_and_rerun(&MOVEC_CLEAR_030);
    let mut bus = CachedCodeBus::new(&words, true);
    let mut cpu = cpu_at_code(CpuType::M68030, CACR_EI | CACR_ED);

    let result = cpu.run_batch(&mut bus, count, &[]);

    assert_eq!(result.exit, BatchExit::BudgetExhausted);
    assert_eq!(result.instructions, count);
    assert_eq!(
        cpu.d(3),
        0x4444_4444,
        "the rerun must fetch the new immediate"
    );
    assert_eq!(cpu.cacr_pending_ops, 0, "the bus consumed the strobes");
    let report = Event::CacheControl {
        cacr: CACR_EI | CACR_ED,
        caar: 0,
        strobes: CACR_CI,
    };
    assert_eq!(bus.cache_control_events(), vec![report, report]);
    // Each report follows the MOVEC's last word and precedes the dbra.
    for at in bus
        .events
        .iter()
        .enumerate()
        .filter(|(_, event)| **event == report)
        .map(|(at, _)| at)
    {
        assert_eq!(bus.events[at - 1], Event::Read(0x1020));
        assert_eq!(bus.events[at + 1], Event::Read(0x1022));
    }
}

#[test]
fn cinv_and_cpush_reach_the_bus_before_the_next_fetch() {
    for cpu_type in [CpuType::M68040, CpuType::M68060] {
        // cinva ic ; cpusha bc
        for (strobe, strobes) in [(0xF498, CACR_CI), (0xF4F8, CACR_CI | CACR_CD)] {
            let (words, count) = rewrite_and_rerun(&[strobe]);
            let mut bus = CachedCodeBus::new(&words, true);
            let cacr = CACR_040_IE | CACR_040_DE;
            let mut cpu = cpu_at_code(cpu_type, cacr);

            let result = cpu.run_batch(&mut bus, count, &[]);

            let label = format!("{cpu_type:?} {strobe:#06X}");
            assert_eq!(result.instructions, count, "{label}");
            assert_eq!(cpu.d(3), 0x4444_4444, "{label}");
            assert_eq!(cpu.cacr_pending_ops, 0, "{label}");
            let report = Event::CacheControl {
                cacr,
                caar: 0,
                strobes,
            };
            assert_eq!(bus.cache_control_events(), vec![report, report], "{label}");
        }
    }
}

/// A bus without the hook sees the strobes pending after the batch, as
/// before; one that declines them leaves them pending too, and is not
/// called again for a strobe that is still pending.
#[test]
fn unconsumed_strobes_stay_pending_for_the_host() {
    let (words, count) = rewrite_and_rerun(&MOVEC_CLEAR_030);
    let mut bus = m68k::LinearMemoryBus::new(0x10000);
    for (i, word) in words.iter().enumerate() {
        bus.load(CODE + 2 * i as u32, &word.to_be_bytes());
    }
    let mut cpu = cpu_at_code(CpuType::M68030, CACR_EI | CACR_ED);
    assert_eq!(cpu.run_batch(&mut bus, count, &[]).instructions, count);
    assert_eq!(cpu.cacr_pending_ops, CACR_CI);

    let mut bus = CachedCodeBus::new(&words, false);
    let mut cpu = cpu_at_code(CpuType::M68030, CACR_EI | CACR_ED);
    assert_eq!(cpu.run_batch(&mut bus, count, &[]).instructions, count);
    assert_eq!(cpu.cacr_pending_ops, CACR_CI);
    assert_eq!(
        bus.cache_control_events(),
        vec![Event::CacheControl {
            cacr: CACR_EI | CACR_ED,
            caar: 0,
            strobes: CACR_CI,
        }]
    );
    assert_eq!(cpu.d(3), 0x3333_3333, "nothing cleared the host cache");
}

/// Only a change is reported: rewriting CACR with its current value is
/// silent, while an enable change without strobes is reported.
#[test]
fn cache_control_reports_changes_only() {
    let words = [
        0x7001, // moveq #1,d0 (EI)
        0x4E7B, 0x0002, // movec d0,cacr (unchanged)
        0x203C, 0x0000, 0x0101, // move.l #(ED|EI),d0
        0x4E7B, 0x0002, // movec d0,cacr (enables the data cache)
        0x60FE, // bra.s *
    ];
    let mut bus = CachedCodeBus::new(&words, true);
    let mut cpu = cpu_at_code(CpuType::M68030, CACR_EI);

    assert_eq!(cpu.run_batch(&mut bus, 4, &[]).instructions, 4);

    assert_eq!(
        bus.cache_control_events(),
        vec![Event::CacheControl {
            cacr: CACR_EI | CACR_ED,
            caar: 0,
            strobes: 0,
        }]
    );
}
