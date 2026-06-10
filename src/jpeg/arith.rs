//! JPEG arithmetic-coded entropy coder (T.81 Annex D + Annex F + Annex H).
//!
//! Implements the Q-coder probability-estimation state machine (Table D.3),
//! the decoder-side Decode(S) / Initdec / Renorm_d / Byte_in / Unstuff_0
//! procedures from Figures D.16–D.22, the encoder-side Code_0(S) /
//! Code_1(S) / Code_MPS / Code_LPS / Renorm_e / Byte_out / Initenc / Flush
//! procedures from Figures D.1–D.15, the JPEG-specific DC-difference and
//! AC-band decoding sequences from Figures F.19–F.24, and the
//! two-dimensional lossless statistical model from §H.1.2.3
//! (L_Context(Da,Db) / X1_Context(Db), Table H.3).
//!
//! Workspace policy bars consulting any external library implementation;
//! the only references used here are the T.81 spec PDF (in
//! `docs/image/jpeg/`) and its Annex K.4 worked example (Tables K.7 / K.8).

use crate::error::{MjpegError as Error, Result};

// ---------------------------------------------------------------------------
// Probability-estimation state machine (Table D.3).
//
// Each entry: (Qe, NLPS, NMPS, Switch_MPS).
//
// Qe — 16-bit interval-fraction estimate for the LPS.
// NLPS — index after an LPS renormalisation.
// NMPS — index after an MPS renormalisation.
// Switch_MPS — when 1 and an LPS occurs, the MPS sense for that context flips.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct QeEntry {
    pub qe: u16,
    pub nlps: u8,
    pub nmps: u8,
    pub switch: u8,
}

const fn e(qe: u16, nlps: u8, nmps: u8, switch: u8) -> QeEntry {
    QeEntry {
        qe,
        nlps,
        nmps,
        switch,
    }
}

/// 113-entry Q-coder probability-estimation table per T.81 Table D.3.
pub const QE_TABLE: [QeEntry; 113] = [
    e(0x5A1D, 1, 1, 1),     // 0
    e(0x2586, 14, 2, 0),    // 1
    e(0x1114, 16, 3, 0),    // 2
    e(0x080B, 18, 4, 0),    // 3
    e(0x03D8, 20, 5, 0),    // 4
    e(0x01DA, 23, 6, 0),    // 5
    e(0x00E5, 25, 7, 0),    // 6
    e(0x006F, 28, 8, 0),    // 7
    e(0x0036, 30, 9, 0),    // 8
    e(0x001A, 33, 10, 0),   // 9
    e(0x000D, 35, 11, 0),   // 10
    e(0x0006, 9, 12, 0),    // 11
    e(0x0003, 10, 13, 0),   // 12
    e(0x0001, 12, 13, 0),   // 13
    e(0x5A7F, 15, 15, 1),   // 14
    e(0x3F25, 36, 16, 0),   // 15
    e(0x2CF2, 38, 17, 0),   // 16
    e(0x207C, 39, 18, 0),   // 17
    e(0x17B9, 40, 19, 0),   // 18
    e(0x1182, 42, 20, 0),   // 19
    e(0x0CEF, 43, 21, 0),   // 20
    e(0x09A1, 45, 22, 0),   // 21
    e(0x072F, 46, 23, 0),   // 22
    e(0x055C, 48, 24, 0),   // 23
    e(0x0406, 49, 25, 0),   // 24
    e(0x0303, 51, 26, 0),   // 25
    e(0x0240, 52, 27, 0),   // 26
    e(0x01B1, 54, 28, 0),   // 27
    e(0x0144, 56, 29, 0),   // 28
    e(0x00F5, 57, 30, 0),   // 29
    e(0x00B7, 59, 31, 0),   // 30
    e(0x008A, 60, 32, 0),   // 31
    e(0x0068, 62, 33, 0),   // 32
    e(0x004E, 63, 34, 0),   // 33
    e(0x003B, 32, 35, 0),   // 34
    e(0x002C, 33, 9, 0),    // 35
    e(0x5AE1, 37, 37, 1),   // 36
    e(0x484C, 64, 38, 0),   // 37
    e(0x3A0D, 65, 39, 0),   // 38
    e(0x2EF1, 67, 40, 0),   // 39
    e(0x261F, 68, 41, 0),   // 40
    e(0x1F33, 69, 42, 0),   // 41
    e(0x19A8, 70, 43, 0),   // 42
    e(0x1518, 72, 44, 0),   // 43
    e(0x1177, 73, 45, 0),   // 44
    e(0x0E74, 74, 46, 0),   // 45
    e(0x0BFB, 75, 47, 0),   // 46
    e(0x09F8, 77, 48, 0),   // 47
    e(0x0861, 78, 49, 0),   // 48
    e(0x0706, 79, 50, 0),   // 49
    e(0x05CD, 48, 51, 0),   // 50
    e(0x04DE, 50, 52, 0),   // 51
    e(0x040F, 50, 53, 0),   // 52
    e(0x0363, 51, 54, 0),   // 53
    e(0x02D4, 52, 55, 0),   // 54
    e(0x025C, 53, 56, 0),   // 55
    e(0x01F8, 54, 57, 0),   // 56
    e(0x01A4, 55, 58, 0),   // 57
    e(0x0160, 56, 59, 0),   // 58
    e(0x0125, 57, 60, 0),   // 59
    e(0x00F6, 58, 61, 0),   // 60
    e(0x00CB, 59, 62, 0),   // 61
    e(0x00AB, 61, 63, 0),   // 62
    e(0x008F, 61, 32, 0),   // 63
    e(0x5B12, 65, 65, 1),   // 64
    e(0x4D04, 80, 66, 0),   // 65
    e(0x412C, 81, 67, 0),   // 66
    e(0x37D8, 82, 68, 0),   // 67
    e(0x2FE8, 83, 69, 0),   // 68
    e(0x293C, 84, 70, 0),   // 69
    e(0x2379, 86, 71, 0),   // 70
    e(0x1EDF, 87, 72, 0),   // 71
    e(0x1AA9, 87, 73, 0),   // 72
    e(0x174E, 72, 74, 0),   // 73
    e(0x1424, 72, 75, 0),   // 74
    e(0x119C, 74, 76, 0),   // 75
    e(0x0F6B, 74, 77, 0),   // 76
    e(0x0D51, 75, 78, 0),   // 77
    e(0x0BB6, 77, 79, 0),   // 78
    e(0x0A40, 77, 48, 0),   // 79
    e(0x5832, 80, 81, 1),   // 80
    e(0x4D1C, 88, 82, 0),   // 81
    e(0x438E, 89, 83, 0),   // 82
    e(0x3BDD, 90, 84, 0),   // 83
    e(0x34EE, 91, 85, 0),   // 84
    e(0x2EAE, 92, 86, 0),   // 85
    e(0x299A, 93, 87, 0),   // 86
    e(0x2516, 86, 71, 0),   // 87
    e(0x5570, 88, 89, 1),   // 88
    e(0x4CA9, 95, 90, 0),   // 89
    e(0x44D9, 96, 91, 0),   // 90
    e(0x3E22, 97, 92, 0),   // 91
    e(0x3824, 99, 93, 0),   // 92
    e(0x32B4, 99, 94, 0),   // 93
    e(0x2E17, 93, 86, 0),   // 94
    e(0x56A8, 95, 96, 1),   // 95
    e(0x4F46, 101, 97, 0),  // 96
    e(0x47E5, 102, 98, 0),  // 97
    e(0x41CF, 103, 99, 0),  // 98
    e(0x3C3D, 104, 100, 0), // 99
    e(0x375E, 99, 93, 0),   // 100
    e(0x5231, 105, 102, 0), // 101
    e(0x4C0F, 106, 103, 0), // 102
    e(0x4639, 107, 104, 0), // 103
    e(0x415E, 103, 99, 0),  // 104
    e(0x5627, 105, 106, 1), // 105
    e(0x50E7, 108, 107, 0), // 106
    e(0x4B85, 109, 103, 0), // 107
    e(0x5597, 110, 109, 0), // 108
    e(0x504F, 111, 107, 0), // 109
    e(0x5A10, 110, 111, 1), // 110
    e(0x5522, 112, 109, 0), // 111
    e(0x59EB, 112, 111, 1), // 112
];

// ---------------------------------------------------------------------------
// Statistics-bin context. Each context tracks an MPS sense (0 or 1) and a
// state index into QE_TABLE. Initial value: mps=0, idx=0 (Qe=0x5A1D).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    pub idx: u8,
    pub mps: u8,
}

// ---------------------------------------------------------------------------
// Bit-source: byte stream with `0xFF 0x00` stuff removal and marker
// detection. Once a real marker is seen, the source pads the C-register
// with zeros (per T.81 D.2.6 Adjust BP / Unstuff_0 footnote).
// ---------------------------------------------------------------------------

pub struct ByteSource<'a> {
    buf: &'a [u8],
    pos: usize,
    /// Once a non-stuff marker is encountered, all subsequent reads return 0
    /// (per T.81 Unstuff_0 — "0-bytes will be fed to the decoder until
    /// decoding is complete"). The marker byte is recorded so the caller
    /// can interpret it (e.g. RSTn for restart synchronisation).
    pub seen_marker: Option<u8>,
}

impl<'a> ByteSource<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            seen_marker: None,
        }
    }

    /// Fetch one decoder byte, applying the `0xFF 0x00` stuff convention.
    /// Returns `Some(byte)` for normal data and `None` once a non-stuff
    /// marker has been seen (after which reads pad with zero).
    pub fn next_byte(&mut self) -> u8 {
        if self.seen_marker.is_some() {
            return 0;
        }
        if self.pos >= self.buf.len() {
            // EOF without a marker — pad with zeros, mark "done".
            self.seen_marker = Some(0);
            return 0;
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        if b == 0xFF {
            // Collapse any run of additional 0xFF fill bytes.
            while self.pos < self.buf.len() && self.buf[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos >= self.buf.len() {
                self.seen_marker = Some(0);
                return 0;
            }
            let nxt = self.buf[self.pos];
            self.pos += 1;
            if nxt == 0x00 {
                // 0xFF 0x00 → literal 0xFF.
                return 0xFF;
            }
            // Real marker.
            self.seen_marker = Some(nxt);
            return 0;
        }
        b
    }
}

// ---------------------------------------------------------------------------
// Q-coder decoder state.
// ---------------------------------------------------------------------------

/// Renormalise the (A, C) registers until A >= 0x8000. Pulls one byte per
/// 8 bits of shift, refilling the low half of C with the next code byte.
pub struct ArithDecoder<'a> {
    pub src: ByteSource<'a>,
    /// Probability-interval register (16-bit logical, but stored as u32).
    pub a: u32,
    /// Code register (32-bit; the high 16 bits Cx + low 16 bits C-low).
    pub c: u32,
    /// Number of valid bits remaining in the low half of C.
    pub ct: i32,
}

impl<'a> ArithDecoder<'a> {
    /// Initdec (Figure D.22): pre-load two bytes into Cx, set A to 0 (logical
    /// 0x10000) and CT to 0 so the first Renorm_d will fetch a fresh byte.
    ///
    /// Per Figure D.22 the procedure is `Byte_in; C = SLL C 8; Byte_in;
    /// C = SLL C 8; CT = 0`. Each Byte_in adds `B << 8` to C, so after the
    /// two byte-in/SLL pairs we have:
    ///   * after first Byte_in:  C = b0 << 8
    ///   * after first SLL:      C = b0 << 16
    ///   * after second Byte_in: C = (b0 << 16) | (b1 << 8)
    ///   * after second SLL:     C = (b0 << 24) | (b1 << 16)
    pub fn new(scan: &'a [u8]) -> Self {
        let mut s = ByteSource::new(scan);
        let b0 = s.next_byte() as u32;
        let b1 = s.next_byte() as u32;
        let c = (b0 << 24) | (b1 << 16);
        Self {
            src: s,
            // The encoder/decoder treat the interval as 0..=0xFFFF logically;
            // 0x0000 here represents the initial 0x10000. Renorm uses the
            // top bit (0x8000) test, which works for both 16- and 17-bit
            // interpretations.
            a: 0,
            c,
            ct: 0,
        }
    }

    /// Re-initialise inside a scan after seeing an RSTn marker. Stats are
    /// reset by the caller; this resets the encoder/decoder synchronisation.
    pub fn restart(&mut self) {
        // ByteSource was exhausted up to the marker; the caller will hand
        // us a fresh slice positioned past the RSTn.
        let b0 = self.src.next_byte() as u32;
        let b1 = self.src.next_byte() as u32;
        self.c = (b0 << 24) | (b1 << 16);
        self.a = 0;
        self.ct = 0;
    }

    /// Replace the byte source (used for restart-interval handling where
    /// the caller positioned a new slice past an RSTn).
    pub fn reset_with_source(&mut self, scan: &'a [u8]) {
        self.src = ByteSource::new(scan);
        let b0 = self.src.next_byte() as u32;
        let b1 = self.src.next_byte() as u32;
        self.c = (b0 << 24) | (b1 << 16);
        self.a = 0;
        self.ct = 0;
    }

    /// Renorm_d (Figure D.19): shift A and C left until A >= 0x8000,
    /// pulling fresh bytes via Byte_in (Figure D.20) when CT runs out.
    fn renorm_d(&mut self) {
        // Loop is "do at least once until A>=0x8000"; here we use a
        // straight while because callers only call renorm when they know
        // A < 0x8000.
        loop {
            if self.ct == 0 {
                let b = self.src.next_byte() as u32;
                self.c = self.c.wrapping_add(b << 8);
                self.ct = 8;
            }
            self.a = (self.a << 1) & 0xFFFF;
            self.c <<= 1;
            self.ct -= 1;
            if self.a >= 0x8000 {
                break;
            }
        }
    }

    /// Decode(S) per Figure D.16 + D.17 + D.18.
    ///
    /// Returns the binary decision (0 or 1). `ctx` is mutated to advance
    /// the probability-state on a renormalisation.
    pub fn decode(&mut self, ctx: &mut Context) -> u8 {
        let entry = QE_TABLE[ctx.idx as usize];
        let qe = entry.qe as u32;
        // A = A - Qe(S)
        self.a = self.a.wrapping_sub(qe) & 0xFFFF;
        // The 16-bit-precision implementation initialises A to 0 instead of
        // 0x10000. Logically A still represents 0x10000 - 0 - ... so the
        // comparison Cx < A uses the LOGICAL value. To make this work, we
        // carry "A is 0 means A is 0x10000" implicitly: the renorm condition
        // (A < 0x8000) is checked on the truncated 16-bit A, but the
        // Cx-vs-A comparison must use a representation that distinguishes
        // 0 from 0x10000. We avoid that by always renormalising as soon as
        // A drops below 0x8000 — at which point the next Decode sees a
        // strictly < 0x10000 value and the comparison works directly.

        let cx = (self.c >> 16) & 0xFFFF;
        if cx < self.a {
            // No conditional exchange: MPS path with A >= 0x8000 → no renorm.
            if self.a < 0x8000 {
                let d = self.cond_mps_exchange(ctx);
                self.renorm_d();
                d
            } else {
                ctx.mps
            }
        } else {
            // Cx >= A: subtract A from Cx and take the LPS path.
            let d = self.cond_lps_exchange(ctx);
            self.renorm_d();
            d
        }
    }

    /// Cond_LPS_exchange (Figure D.17). On entry Cx >= A.
    fn cond_lps_exchange(&mut self, ctx: &mut Context) -> u8 {
        let entry = QE_TABLE[ctx.idx as usize];
        let qe = entry.qe as u32;
        // Cx = Cx - A   (subtract MPS sub-interval base from code register)
        // A  = Qe(S)
        let cx = (self.c >> 16) & 0xFFFF;
        let new_cx = cx.wrapping_sub(self.a) & 0xFFFF;
        self.c = (self.c & 0xFFFF) | (new_cx << 16);
        let was_a = self.a;
        self.a = qe;

        let d;
        if was_a < qe {
            // The MPS sub-interval was actually smaller than the LPS sub-
            // interval. The encoder swapped them, so what looked like an
            // LPS event is really an MPS.
            d = ctx.mps;
            // Estimate_Qe_after_MPS.
            ctx.idx = entry.nmps;
        } else {
            // Genuine LPS event.
            d = 1 - ctx.mps;
            // Estimate_Qe_after_LPS (and switch MPS sense if the table says so).
            if entry.switch == 1 {
                ctx.mps ^= 1;
            }
            ctx.idx = entry.nlps;
        }
        d
    }

    /// Cond_MPS_exchange (Figure D.18). Reached when A < 0x8000 after the
    /// MPS-side decision (Cx < A).
    fn cond_mps_exchange(&mut self, ctx: &mut Context) -> u8 {
        let entry = QE_TABLE[ctx.idx as usize];
        let qe = entry.qe as u32;
        let d;
        if self.a < qe {
            // Conditional exchange: code path swaps interpretation.
            d = 1 - ctx.mps;
            if entry.switch == 1 {
                ctx.mps ^= 1;
            }
            ctx.idx = entry.nlps;
        } else {
            d = ctx.mps;
            ctx.idx = entry.nmps;
        }
        d
    }

    /// Has the source consumed any non-stuff marker yet? Returns the marker
    /// byte if so (for restart synchronisation).
    pub fn marker(&self) -> Option<u8> {
        self.src.seen_marker
    }
}

// ---------------------------------------------------------------------------
// Q-coder encoder state (T.81 §D.1).
// ---------------------------------------------------------------------------

/// Arithmetic entropy encoder per T.81 Annex D §D.1: Initenc (Figure D.12),
/// Code_MPS / Code_LPS with conditional exchange (Figures D.3 / D.4),
/// Renorm_e (Figure D.7), Byte_out with carry resolution + `0xFF 0x00`
/// stuffing (Figures D.8–D.11), and Flush / Clear_final_bits /
/// Discard_final_zeros termination (Figures D.13–D.15).
pub struct ArithEncoder {
    out: Vec<u8>,
    /// Probability-interval register. Initialised to the full X'10000'
    /// (Figure D.12); always > X'7FFF' between events.
    a: u32,
    /// Code register (Figure D.2 layout: output bits surface at C >> 19).
    c: u32,
    /// Bit counter until the next Byte_out. Initenc sets 11 — three spacer
    /// bits plus eight output bits (§D.1.7).
    ct: i32,
    /// Stack counter ST — number of X'FF' bytes withheld pending carry
    /// resolution (Figure D.8).
    st: u32,
}

impl ArithEncoder {
    /// Initenc (Figure D.12): A = X'10000', C = 0, CT = 11, ST = 0.
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            a: 0x10000,
            c: 0,
            ct: 11,
            st: 0,
        }
    }

    /// Code one binary decision `d` (0 or 1) against the statistics bin
    /// `ctx`. This is Code_1(S) / Code_0(S) (Figures D.1 / D.2) folded into
    /// one entry point: `d == MPS(S)` takes the Code_MPS path (Figure D.4),
    /// otherwise Code_LPS (Figure D.3).
    pub fn code_bit(&mut self, ctx: &mut Context, d: u8) {
        let entry = QE_TABLE[ctx.idx as usize];
        let qe = entry.qe as u32;
        if d == ctx.mps {
            // Code_MPS (Figure D.4): A = A - Qe; renormalise only when the
            // interval drops below X'8000', applying conditional exchange
            // when the MPS sub-interval came out smaller than Qe.
            self.a -= qe;
            if self.a < 0x8000 {
                if self.a < qe {
                    self.c += self.a;
                    self.a = qe;
                }
                ctx.idx = entry.nmps;
                self.renorm_e();
            }
        } else {
            // Code_LPS (Figure D.3): A = A - Qe; without conditional
            // exchange the LPS takes the upper sub-interval (C += A,
            // A = Qe); with exchange (A < Qe) the LPS keeps the lower,
            // now-smaller MPS sub-interval.
            self.a -= qe;
            if self.a >= qe {
                self.c += self.a;
                self.a = qe;
            }
            if entry.switch == 1 {
                ctx.mps ^= 1;
            }
            ctx.idx = entry.nlps;
            self.renorm_e();
        }
    }

    /// Renorm_e (Figure D.7): shift A and C left until A >= X'8000',
    /// emitting a byte via Byte_out every 8 shifts.
    fn renorm_e(&mut self) {
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byte_out();
                self.ct = 8;
            }
            if self.a >= 0x8000 {
                break;
            }
        }
    }

    /// Byte_out (Figure D.8) with Output_stacked_zeros (D.9),
    /// Output_stacked_X'FF's (D.10) and Stuff_0 (D.11).
    fn byte_out(&mut self) {
        let t = self.c >> 19;
        if t > 0xFF {
            // Carry-over: add the carry to the previous output byte, then
            // stuff a zero if that made it X'FF' (Stuff_0), then convert
            // any stacked X'FF' bytes to zeros (the carry rippled through
            // them) before emitting the new byte.
            if let Some(last) = self.out.last_mut() {
                *last = last.wrapping_add(1);
                if *last == 0xFF {
                    self.out.push(0);
                }
            }
            for _ in 0..self.st {
                self.out.push(0);
            }
            self.st = 0;
            self.out.push((t & 0xFF) as u8);
        } else if t == 0xFF {
            // Withhold X'FF' bytes until the carry is resolved (ST += 1).
            self.st += 1;
        } else {
            // Carry resolved: flush stacked X'FF's, each followed by a
            // stuffed zero, then the fresh byte.
            for _ in 0..self.st {
                self.out.push(0xFF);
                self.out.push(0);
            }
            self.st = 0;
            self.out.push(t as u8);
        }
        self.c &= 0x7FFFF;
    }

    /// Flush (Figure D.13): Clear_final_bits (Figure D.14), two aligned
    /// Byte_outs, then Discard_final_zeros (Figure D.15). Returns the
    /// completed entropy-coded segment (the caller appends the following
    /// marker; the decoder pads with zero bits at the marker per §D.1.8).
    pub fn finish(mut self) -> Vec<u8> {
        // Clear_final_bits: zero as many low-order C bits as possible
        // without leaving the final interval [C, C + A).
        let mut t = self.c.wrapping_add(self.a - 1) & 0xFFFF_0000;
        if t < self.c {
            t += 0x8000;
        }
        self.c = t;
        // Align + emit the last two bytes.
        self.c <<= self.ct as u32;
        self.byte_out();
        self.c <<= 8;
        self.byte_out();
        // Any X'FF' bytes still stacked can no longer receive a carry —
        // they are real data bytes and each needs its stuffed zero.
        for _ in 0..self.st {
            self.out.push(0xFF);
            self.out.push(0);
        }
        self.st = 0;
        // Discard_final_zeros: trailing zero bytes not preceded by X'FF'
        // may be dropped; a zero following X'FF' is a stuffed zero and
        // shall be kept.
        while self.out.last() == Some(&0) {
            self.out.pop();
        }
        if self.out.last() == Some(&0xFF) {
            self.out.push(0);
        }
        self.out
    }
}

impl Default for ArithEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Statistics layout (T.81 F.1.4.4). Each component / scan owns one DC area
// and one AC area (both shared across blocks of the component).
// ---------------------------------------------------------------------------

/// DC statistics area. 49 bins per F.1.4.4.1.3:
///   * 5 sets of 4 bins (0/4/8/12/16 base for S0/SS/SP/SN per Da class),
///   * 14 X1..X15 magnitude bins starting at index 20,
///   * 14 M2..M15 magnitude-bit bins starting at index 21+13 = 34.
///
/// The DC-context-base S0 is computed from the previous-block DC difference
/// `Da` (see `dc_context`).
#[derive(Clone, Debug)]
pub struct DcStats {
    pub bins: [Context; 49],
    /// Conditioning lower bound — see DAC marker, default L=0.
    pub l: u8,
    /// Conditioning upper bound — see DAC marker, default U=1.
    pub u: u8,
    /// `Prev` — the DC difference from the previous block of this component
    /// (zero at scan start and after each restart per F.1.4.4.1.5).
    pub prev_diff: i32,
    /// `Pred` — the DC value from the previous block. Distinct from
    /// `prev_diff` because predictions accumulate.
    pub pred: i32,
}

impl DcStats {
    pub fn new() -> Self {
        Self {
            bins: [Context::default(); 49],
            l: 0,
            u: 1,
            prev_diff: 0,
            pred: 0,
        }
    }

    /// Reset learning + DC predictor (called at scan start and at each RSTn).
    pub fn reset(&mut self) {
        for b in self.bins.iter_mut() {
            *b = Context::default();
        }
        self.prev_diff = 0;
        self.pred = 0;
    }

    /// Reset only the per-restart predictor state (DC predictor + Da). Stats
    /// are also reset per spec at restart points.
    pub fn restart_reset(&mut self) {
        // Per F.2.4.4 + F.1.4.4.1.5, statistics are also re-initialised at
        // restart, matching the encoder's Initdec call.
        for b in self.bins.iter_mut() {
            *b = Context::default();
        }
        self.prev_diff = 0;
        self.pred = 0;
    }

    /// DC_Context(Da): map the previous-block difference to a base bin
    /// (0, 4, 8, 12, 16 for zero/small+/small-/large+/large-).
    pub fn dc_context(&self) -> usize {
        let da = self.prev_diff;
        let lower = if self.l == 0 {
            0
        } else {
            1i32 << (self.l - 1) as i32
        };
        let upper = 1i32 << self.u as i32;
        let abs = da.unsigned_abs() as i32;
        if abs <= lower {
            0
        } else if abs <= upper {
            if da > 0 {
                4
            } else {
                8
            }
        } else if da > 0 {
            12
        } else {
            16
        }
    }
}

impl Default for DcStats {
    fn default() -> Self {
        Self::new()
    }
}

/// AC statistics area. Per F.1.4.4.2 the layout is contiguous:
///   * For each K in 1..=63: 3 bins SE/S0/SP_or_SN starting at base 3*(K-1).
///     SS (sign) uses a fixed 0x5A1D estimate, NOT a tracked bin.
///   * X2..X15 magnitude bins for low-K (K <= Kx) starting at index 189.
///   * X2..X15 magnitude bins for high-K (K > Kx) starting at index 217.
///   * M2..M15 magnitude-bit bins for low-K starting at 189+14 = 203.
///   * M2..M15 magnitude-bit bins for high-K starting at 217+14 = 231.
///
/// Spec gives the largest SP value as 188 (3*(63-1) + 2 = 188), so the
/// SE/S0/SP region uses 0..=188 = 189 bins. Adding X2..M15 for each side
/// (28 bins each, 2 sides) brings the total to 189 + 56 = 245 bins.
#[derive(Clone, Debug)]
pub struct AcStats {
    pub bins: [Context; 245],
    /// Conditioning Kx threshold — default 5.
    pub kx: u8,
}

impl AcStats {
    pub fn new() -> Self {
        Self {
            bins: [Context::default(); 245],
            kx: 5,
        }
    }

    pub fn reset(&mut self) {
        for b in self.bins.iter_mut() {
            *b = Context::default();
        }
    }

    pub fn restart_reset(&mut self) {
        self.reset();
    }
}

impl Default for AcStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics area for subsequent (refinement) successive-approximation
/// scans of AC coefficients — T.81 §G.1.3.3.1 / Table G.2: a contiguous
/// set of 189 bins, three per coefficient index `K ∈ 1..=63`:
///   * `SE = 3 × (K − 1)` — the `K = EOB` (end-of-band) decision;
///   * `S0 = SE + 1` — the `V = 0` decision;
///   * `SC = S0 + 1` — the `LSB ZZ(K) = 1` correction-bit decision.
///
/// The sign of a coefficient with magnitude one (newly nonzero at this
/// precision) uses the fixed 0x5A1D estimate, not a tracked bin. No
/// magnitude tree exists in refinement scans, so there is no Kx
/// conditioning either.
#[derive(Clone, Debug)]
pub struct AcRefineStats {
    pub bins: [Context; 189],
}

impl AcRefineStats {
    pub fn new() -> Self {
        Self {
            bins: [Context::default(); 189],
        }
    }

    pub fn reset(&mut self) {
        for b in self.bins.iter_mut() {
            *b = Context::default();
        }
    }

    pub fn restart_reset(&mut self) {
        self.reset();
    }
}

impl Default for AcRefineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Lossless (SOF11) statistics area — 158 bins per T.81 §H.1.2.3.2:
///   * 25 sets of four bins (S0 / SS / SP / SN) at bases 0, 4, ..., 96,
///     selected by `L_Context(Da, Db)` over the 5 × 5 conditioning array
///     of Figure H.2 (rows = difference-to-left class, columns =
///     difference-above class, each ordered zero / +S / −S / +L / −L);
///   * two 29-bin magnitude sets (X1..X15 then M2..M15, with
///     `Mn = Xn + 14` per Table H.3) at base 100 when Db classifies as
///     zero / small and base 129 when Db classifies as large
///     (`X1_Context(Db)`, §H.1.2.3.2).
///
/// The L / U conditioning bounds default to (0, 1) per §H.1.2.3.3 and may
/// be overridden by a DAC marker segment (Tc = 0 entry).
#[derive(Clone, Debug)]
pub struct LosslessStats {
    pub bins: [Context; 158],
    /// Conditioning lower bound L (DAC override; default 0).
    pub l: u8,
    /// Conditioning upper bound U (DAC override; default 1).
    pub u: u8,
}

impl LosslessStats {
    pub fn new() -> Self {
        Self {
            bins: [Context::default(); 158],
            l: 0,
            u: 1,
        }
    }

    /// Re-initialise every bin (scan start and each restart interval, per
    /// §H.1.2.3.4 / Annex D defaults).
    pub fn reset(&mut self) {
        for b in self.bins.iter_mut() {
            *b = Context::default();
        }
    }

    /// Classify a neighbouring difference into the five F.1.4.4.1.2
    /// categories using the L / U bounds: 0 = zero, 1 = small positive,
    /// 2 = small negative, 3 = large positive, 4 = large negative. The
    /// lower (exclusive) bound for "small" is 0 when L = 0 and 2^(L−1)
    /// otherwise; the upper (inclusive) bound is 2^U.
    pub fn classify(&self, d: i32) -> usize {
        let lower = if self.l == 0 {
            0
        } else {
            1i32 << (self.l - 1) as i32
        };
        let upper = 1i32 << self.u as i32;
        let abs = d.unsigned_abs() as i32;
        if abs <= lower {
            0
        } else if abs <= upper {
            if d > 0 {
                1
            } else {
                2
            }
        } else if d > 0 {
            3
        } else {
            4
        }
    }

    /// `L_Context(Da, Db)` — base S0 bin from the Figure H.2 array
    /// (row stride 20 over the left-difference class, column stride 4
    /// over the above-difference class).
    fn s0(&self, da: i32, db: i32) -> usize {
        20 * self.classify(da) + 4 * self.classify(db)
    }

    /// `X1_Context(Db)` — magnitude-set base: 100 when Db is zero /
    /// small (either sign), 129 when Db is large (§H.1.2.3.2).
    fn x1(&self, db: i32) -> usize {
        if self.classify(db) <= 2 {
            100
        } else {
            129
        }
    }
}

impl Default for LosslessStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JPEG arithmetic block decode (DC + AC zigzag).
// ---------------------------------------------------------------------------

/// Fixed 50/50 estimator decision: always Qe = 0x5A1D, MPS = 0, never
/// adapts on renormalisation. Implemented by decoding against a throwaway
/// context that we reset before every call. T.81 uses this for the sign
/// of DC differences / AC coefficients (Tables F.4 / F.5), the sign of
/// newly-nonzero coefficients in AC refinement scans (§G.1.3.3), and the
/// DC least-significant-bit decisions of successive-approximation
/// refinement scans (§G.1.3.1).
pub fn decode_fixed_bit(d: &mut ArithDecoder<'_>) -> u8 {
    let mut ctx = Context { idx: 0, mps: 0 };
    d.decode(&mut ctx)
}

/// Decode the magnitude category and exact bits per F.2.4.3.1.2 / .3.
/// `is_dc` selects the DC vs AC bin layout (Tables F.4 vs F.5). `ac_ctx_low`
/// is true when the AC sub-band index K satisfies K <= Kx (selects the
/// low-K X2..X15/M2..M15 group). Returns the absolute magnitude (Sz + 1).
fn decode_magnitude(
    d: &mut ArithDecoder<'_>,
    bins: &mut [Context],
    sx_first: usize,
    x1_base: usize,
    is_dc: bool,
    ac_ctx_low: bool,
) -> Result<i32> {
    // First magnitude decision: Sz < 1?
    // For DC, the first decision uses SP or SN (sx_first), bin already
    // selected by Decode_sign_of_V. For AC, both SP and SN coincide and
    // sit at S0 + 1 = SE + 2 (sx_first).
    let mut s = sx_first;
    let mut m: i32 = 1;
    let mut category: u32 = 0;

    // Decision tree from Figure F.23.
    // First decision: is Sz >= 1?
    let d0 = d.decode(&mut bins[s]);
    if d0 != 0 {
        // Sz >= 1, advance.
        m = 2;
        s = x1_base;
        category = 1;
        // Sz >= 2?
        let d1 = d.decode(&mut bins[s]);
        if d1 != 0 {
            m = 4;
            // X2 base depends on DC vs AC.
            s = if is_dc {
                x1_base + 1 // X2 = X1 + 1
            } else if ac_ctx_low {
                189
            } else {
                217
            };
            category = 2;
            // Loop: decode Sz < M? for M = 4, 8, 16, ...
            loop {
                let dx = d.decode(&mut bins[s]);
                if dx == 0 {
                    // Magnitude in [M/2 .. M-1]; M is the exclusive bound.
                    break;
                }
                m <<= 1;
                s += 1;
                category += 1;
                if m == 0 || category > 15 {
                    return Err(Error::invalid("arith: magnitude category > 15"));
                }
            }
        }
    }

    // After the magnitude category is fixed, decode_sz_bits (Figure F.24).
    // M is the exclusive upper bound. SRL M 1 makes it the leading-bit
    // mask. Then for each lower-order bit (bit positions M/2 down to 1),
    // decode against a magnitude-bit bin.
    //
    // The "S = S + 14" step from Figure F.24 sets the magnitude-bit bin.
    // For DC: M2..M15 start at X1 + 14 = base+14, X3+14, ..., X15+14.
    // For AC: M2..M15 start at X2 + 14 (= 189+14 or 217+14).
    let mut sz: i32 = 0;
    if category >= 2 {
        // The S used to decode the leading "1" of the magnitude is the
        // last X-bin we read a 0 from (S above). Add 14 to land on the
        // matching M-bin. But for category==2, the bit-reader uses M2 — the
        // X2 bin's M-shadow.
        let m_bin_base = s + 14;
        // Determine bit-mask: M is the exclusive upper bound (leading bit set).
        // The leading "1" implied by the category is already accounted for;
        // we need (category - 1) low-order bits.
        let mut bit = m >> 1;
        // Implicit leading 1.
        sz = bit;
        bit >>= 1;
        while bit != 0 {
            let b = d.decode(&mut bins[m_bin_base]);
            if b != 0 {
                sz |= bit;
            }
            bit >>= 1;
        }
    } else if category == 1 {
        // Sz < 2 → Sz must be 1 exactly (since we already established Sz >= 1).
        sz = 1;
    }

    Ok(sz)
}

/// Decode one DC difference for one block. Returns the signed difference
/// to ADD to the predictor (caller updates `pred`).
pub fn decode_dc_diff(d: &mut ArithDecoder<'_>, dc: &mut DcStats) -> Result<i32> {
    // S0 = base bin from Da classification.
    let s0 = dc.dc_context();
    let zero = d.decode(&mut dc.bins[s0]);
    if zero == 0 {
        // DIFF = 0; Da updates to 0.
        dc.prev_diff = 0;
        return Ok(0);
    }
    // DIFF != 0 → decode sign + magnitude.
    let sign = d.decode(&mut dc.bins[s0 + 1]);
    let sx_first = if sign == 0 { s0 + 2 } else { s0 + 3 };
    // X1 base for DC is index 20 (per Table F.4). M-bins shadow X bins at +14.
    let mag = decode_magnitude(d, &mut dc.bins, sx_first, 20, true, false)?;
    let v = mag + 1;
    let signed = if sign != 0 { -v } else { v };
    dc.prev_diff = signed;
    Ok(signed)
}

/// Decode the AC band [ss..=se] for one block. Writes into `coefs` in
/// natural order (caller pre-zeroed). `coefs` must be len 64.
pub fn decode_ac(
    d: &mut ArithDecoder<'_>,
    ac: &mut AcStats,
    coefs: &mut [i32; 64],
    ss: usize,
    se: usize,
) -> Result<()> {
    use crate::jpeg::zigzag::ZIGZAG;
    let mut k = ss;
    loop {
        // SE bin tracks current K (re-evaluated each iteration since K
        // advances through both the EOB-test loop and the zero-run loop).
        let se_bin = 3 * (k - 1);
        let eob = d.decode(&mut ac.bins[se_bin]);
        if eob != 0 {
            return Ok(());
        }
        // Inner zero-run loop: V == 0 decision uses S0 = SE + 1.
        // Each zero advances K, which in turn advances S0 (and SE for the
        // EOB-decision in the next outer iteration).
        loop {
            let s0 = 3 * (k - 1) + 1;
            let zero = d.decode(&mut ac.bins[s0]);
            if zero != 0 {
                break;
            }
            k += 1;
            if k > se {
                return Err(Error::invalid("arith AC: run past Se"));
            }
        }
        // Sign uses the fixed 0x5A1D bin per Table F.5.
        let sign = decode_fixed_bit(d);
        // SP / SN both = S0 + 1 = SE + 2.
        let sx_first = 3 * (k - 1) + 2;
        let ac_low = (k as u8) <= ac.kx;
        // X1 for AC equals S0 + 1 (= SE + 2) per Table F.5; X2 base differs
        // between low/high K.
        let mag = decode_magnitude(d, &mut ac.bins, sx_first, sx_first, false, ac_low)?;
        let v = mag + 1;
        let signed = if sign != 0 { -v } else { v };
        coefs[ZIGZAG[k]] = signed;
        if k == se {
            return Ok(());
        }
        k += 1;
    }
}

/// Decode one block's AC refinement (a subsequent successive-approximation
/// scan) over the band `[ss..=se]` — the decoder mirror of T.81
/// Figures G.10 / G.11 with the Table G.2 context indices:
///
///   * `EOBx`, the end-of-band index left by the previous scan of this
///     component (§G.1.3.3), is recovered from the coefficient history —
///     the index following the last nonzero coefficient in the band. The
///     end-of-band decision is bypassed while `K < EOBx` and after coding
///     a coefficient at `K = Se`.
///   * A zero-history coefficient decodes a `V = 0` decision (`S0`);
///     when it signals nonzero, a fixed-estimate sign follows and the
///     coefficient becomes `±2^Al`.
///   * A nonzero-history coefficient decodes one correction bit (`SC`)
///     which, when set, grows the magnitude by `2^Al`.
///
/// Once the `K = EOB` decision signals end-of-band, the rest of the band
/// is unchanged — unlike the Huffman refinement procedure there are no
/// trailing correction bits, because the arithmetic EOB is defined as the
/// position following the last nonzero coefficient at the *current*
/// precision (§G.1.3.2 NOTE), which is never below `EOBx`.
pub fn decode_ac_refine(
    d: &mut ArithDecoder<'_>,
    st: &mut AcRefineStats,
    coefs: &mut [i32; 64],
    ss: usize,
    se: usize,
    al: u8,
) -> Result<()> {
    use crate::jpeg::zigzag::ZIGZAG;
    if ss == 0 || se > 63 || ss > se {
        return Err(Error::invalid("arith AC refine: invalid band"));
    }
    let p1: i32 = 1 << al;
    // EOBx: index following the last coefficient already nonzero from the
    // prior scan(s) of this component, within the band.
    let mut eobx = ss;
    for k in ss..=se {
        if coefs[ZIGZAG[k]] != 0 {
            eobx = k + 1;
        }
    }
    let mut k = ss;
    loop {
        if k >= eobx {
            let se_bin = 3 * (k - 1);
            if d.decode(&mut st.bins[se_bin]) != 0 {
                // End-of-band: every remaining coefficient keeps its
                // current value (all zero-history past EOBx).
                return Ok(());
            }
        }
        // Walk to the next coefficient coded in this scan: a correction
        // bit for nonzero history, or a fresh ±1 at bit position Al.
        loop {
            let pos = ZIGZAG[k];
            if coefs[pos] != 0 {
                let t = d.decode(&mut st.bins[3 * (k - 1) + 2]);
                if t != 0 {
                    if coefs[pos] >= 0 {
                        coefs[pos] += p1;
                    } else {
                        coefs[pos] -= p1;
                    }
                }
                break;
            }
            if d.decode(&mut st.bins[3 * (k - 1) + 1]) != 0 {
                let sign = decode_fixed_bit(d);
                coefs[pos] = if sign != 0 { -p1 } else { p1 };
                break;
            }
            k += 1;
            if k > se {
                return Err(Error::invalid("arith AC refine: run past Se"));
            }
        }
        if k == se {
            return Ok(());
        }
        k += 1;
    }
}

/// Decode one lossless prediction difference per the §H.1.2.3 statistical
/// model (Table H.3 decision tree — same shape as the DC tree of Table F.4,
/// conditioned on the differences `da` (sample to the left) and `db` (sample
/// in the line above) instead of the previous-block DC difference). T.81
/// §H.2.2 specifies the decoder mirrors the §H.1.2 coding model; the decoded
/// signed difference is added modulo 2^16 to the prediction by the caller.
pub fn decode_lossless_diff(
    d: &mut ArithDecoder<'_>,
    st: &mut LosslessStats,
    da: i32,
    db: i32,
) -> Result<i32> {
    let s0 = st.s0(da, db);
    let zero = d.decode(&mut st.bins[s0]);
    if zero == 0 {
        return Ok(0);
    }
    // SS = S0 + 1 (sign), then SP = S0 + 2 / SN = S0 + 3 for the first
    // magnitude decision, per Table H.3.
    let sign = d.decode(&mut st.bins[s0 + 1]);
    let sx_first = s0 + 2 + sign as usize;
    let x1 = st.x1(db);
    // The magnitude tree uses the DC layout (X2 = X1 + 1, Mn = Xn + 14).
    let mag = decode_magnitude(d, &mut st.bins, sx_first, x1, true, false)?;
    let v = mag + 1;
    Ok(if sign != 0 { -v } else { v })
}

/// Encode one lossless prediction difference — the encoder-side mirror of
/// [`decode_lossless_diff`] (T.81 §H.1.2.3, Table H.3). `v` must be the
/// modulo-2^16 difference reduced to the representative range
/// `-32768..=32767` (magnitude `Sz = |v| - 1` caps at 32767, the last
/// `Sz < 2^15` decision of Table H.3).
pub fn encode_lossless_diff(
    e: &mut ArithEncoder,
    st: &mut LosslessStats,
    da: i32,
    db: i32,
    v: i32,
) -> Result<()> {
    let s0 = st.s0(da, db);
    if v == 0 {
        e.code_bit(&mut st.bins[s0], 0);
        return Ok(());
    }
    e.code_bit(&mut st.bins[s0], 1);
    let sign = u8::from(v < 0);
    e.code_bit(&mut st.bins[s0 + 1], sign);
    let sx_first = s0 + 2 + sign as usize;
    let sz = v.unsigned_abs() - 1;
    if sz > 32767 {
        return Err(Error::invalid("arith lossless: |diff| > 32768"));
    }
    let x1 = st.x1(db);
    encode_magnitude(e, &mut st.bins, sx_first, x1, sz);
    Ok(())
}

/// Encoder-side mirror of [`decode_magnitude`] for the DC-shaped bin layout
/// (X2 = X1 + 1, M-bins shadow X-bins at +14): the magnitude-category
/// decision sequence of Figure F.8 (code_log2_Sz) followed by the magnitude
/// bit pattern of Figure F.9 (code_Sz_bits).
/// `sz` (= magnitude − 1) must be <= 32767, which the callers enforce.
/// `pub(crate)` so the SOF10 round-trip test scaffolding in `decoder.rs`
/// can drive the DC statistical model from the encode side.
pub(crate) fn encode_magnitude(
    e: &mut ArithEncoder,
    bins: &mut [Context],
    sx_first: usize,
    x1: usize,
    sz: u32,
) {
    // First decision: Sz >= 1?
    if sz == 0 {
        e.code_bit(&mut bins[sx_first], 0);
        return;
    }
    e.code_bit(&mut bins[sx_first], 1);
    // Second decision (X1): Sz >= 2?
    if sz < 2 {
        e.code_bit(&mut bins[x1], 0);
        return;
    }
    e.code_bit(&mut bins[x1], 1);
    // Category walk: at bin Xn, code 1 while Sz >= 2^n. Sz <= 32767
    // guarantees termination at X15.
    let mut m: u32 = 4;
    let mut s = x1 + 1; // X2
    while sz >= m {
        e.code_bit(&mut bins[s], 1);
        m <<= 1;
        s += 1;
    }
    e.code_bit(&mut bins[s], 0);
    // Magnitude bits below the implicit leading 1 (M-bin = X-bin + 14).
    let m_bin = s + 14;
    let mut bit = m >> 2;
    while bit != 0 {
        e.code_bit(&mut bins[m_bin], u8::from(sz & bit != 0));
        bit >>= 1;
    }
}

// ---------------------------------------------------------------------------
// Tests against the K.4 example trace (T.81 Annex K).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The K.4.1 source bit sequence (decoder input bytes from Table K.8 /
    /// the encoder output listed at the top of K.4.1).
    const K4_ENCODED: [u8; 32] = [
        0x65, 0x5B, 0x51, 0x44, 0xF7, 0x96, 0x9D, 0x51, 0x78, 0x55, 0xBF, 0xFF, 0x00, 0xFC, 0x51,
        0x84, 0xC7, 0xCE, 0xF9, 0x39, 0x00, 0x28, 0x7D, 0x46, 0x70, 0x8E, 0xCB, 0xC0, 0xF6, 0xFF,
        0xD9, 0x00,
    ];

    /// Source plaintext (256-bit input the encoder consumed).
    const K4_PLAINTEXT: [u8; 32] = [
        0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00, 0xC0, 0x03, 0x52, 0x87, 0x2A, 0xAA, 0xAA, 0xAA,
        0xAA, 0x82, 0xC0, 0x20, 0x00, 0xFC, 0xD7, 0x9E, 0xF6, 0x74, 0xEA, 0xAB, 0xF7, 0x69, 0x7E,
        0xE7, 0x4C,
    ];

    #[test]
    fn k4_arith_roundtrip_decode() {
        // Decode 256 bits using a single context (the encoder used a single
        // context too — that's the whole point of the K.4 test).
        // Note: the encoded stream uses the standard 0xFF-stuff convention,
        // and the trailing 0xFF 0xD9 is the EOI marker; ByteSource will
        // notice it and pad with zeros.
        // The encoded slice we feed is everything BEFORE 0xFF 0xD9 (offsets
        // 0..29 in K4_ENCODED).
        let scan = &K4_ENCODED[..29];
        let mut d = ArithDecoder::new(scan);
        let mut ctx = Context::default();
        let mut decoded = vec![0u8; 32];
        for byte_i in 0..32 {
            let mut byte = 0u8;
            for _ in 0..8 {
                let b = d.decode(&mut ctx);
                byte = (byte << 1) | b;
            }
            decoded[byte_i] = byte;
        }
        assert_eq!(
            decoded, K4_PLAINTEXT,
            "K.4 arithmetic decode did not reproduce the source plaintext"
        );
    }

    #[test]
    fn qe_table_initial_state() {
        assert_eq!(QE_TABLE[0].qe, 0x5A1D);
        assert_eq!(QE_TABLE[0].nlps, 1);
        assert_eq!(QE_TABLE[0].nmps, 1);
        assert_eq!(QE_TABLE[0].switch, 1);
        assert_eq!(QE_TABLE.len(), 113);
    }

    /// Encoder counterpart of the K.4.1 test: coding the 256-bit test
    /// sequence with a single context must reproduce the spec's listed
    /// compressed data sequence (the entropy-coded segment portion, i.e.
    /// everything before the appended EOI marker).
    #[test]
    fn k4_arith_roundtrip_encode() {
        let mut e = ArithEncoder::new();
        let mut ctx = Context::default();
        for &byte in K4_PLAINTEXT.iter() {
            for bit in (0..8).rev() {
                e.code_bit(&mut ctx, (byte >> bit) & 1);
            }
        }
        let out = e.finish();
        assert_eq!(
            out,
            &K4_ENCODED[..29],
            "K.4 arithmetic encode did not reproduce the spec's compressed sequence"
        );
    }

    /// Self-consistency: a pseudorandom decision stream coded across
    /// several independent contexts must decode back bit-exact.
    #[test]
    fn arith_encode_decode_roundtrip_multi_context() {
        // xorshift32 — deterministic, no external dep.
        let mut s = 0x1234_5678u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        const N: usize = 4096;
        const NCTX: usize = 7;
        let mut bits = Vec::with_capacity(N);
        let mut ctx_ids = Vec::with_capacity(N);
        for _ in 0..N {
            let r = next();
            // Skewed bit distribution so contexts adapt away from 50/50.
            bits.push(u8::from((r & 0x7) == 0));
            ctx_ids.push(((r >> 8) as usize) % NCTX);
        }
        let mut e = ArithEncoder::new();
        let mut ectx = [Context::default(); NCTX];
        for i in 0..N {
            e.code_bit(&mut ectx[ctx_ids[i]], bits[i]);
        }
        let coded = e.finish();
        let mut d = ArithDecoder::new(&coded);
        let mut dctx = [Context::default(); NCTX];
        for i in 0..N {
            assert_eq!(
                d.decode(&mut dctx[ctx_ids[i]]),
                bits[i],
                "decision {i} mismatched"
            );
        }
    }

    /// Lossless-diff layer: encode a difference stream with realistic
    /// Da / Db conditioning and decode it back, across default and
    /// DAC-overridden (L, U) bounds.
    #[test]
    fn lossless_diff_roundtrip() {
        for (l, u) in [(0u8, 1u8), (2, 5), (0, 15)] {
            let diffs: Vec<i32> = vec![
                0, 1, -1, 2, -2, 3, 0, 0, 17, -300, 4095, -4096, 32767, -32768, 0, 5, 12345,
                -12345, 1, 0, -1, 255, -256, 32767, -32768, 8, 0,
            ];
            let mut est = LosslessStats::new();
            est.l = l;
            est.u = u;
            let mut e = ArithEncoder::new();
            let mut da = 0i32;
            // Db cycles over earlier diffs to exercise every context class.
            for (i, &v) in diffs.iter().enumerate() {
                let db = if i >= 5 { diffs[i - 5] } else { 0 };
                encode_lossless_diff(&mut e, &mut est, da, db, v).unwrap();
                da = v;
            }
            let coded = e.finish();
            let mut dst = LosslessStats::new();
            dst.l = l;
            dst.u = u;
            let mut d = ArithDecoder::new(&coded);
            let mut da = 0i32;
            for (i, &v) in diffs.iter().enumerate() {
                let db = if i >= 5 { diffs[i - 5] } else { 0 };
                let got = decode_lossless_diff(&mut d, &mut dst, da, db).unwrap();
                assert_eq!(got, v, "diff {i} mismatched at L={l} U={u}");
                da = v;
            }
        }
    }

    /// The Figure H.2 conditioning array bases and the X1_Context split.
    #[test]
    fn lossless_context_bases() {
        let st = LosslessStats::new(); // L = 0, U = 1
                                       // Default bounds: zero = {0}, small = |d| <= 2, large = |d| > 2.
        assert_eq!(st.classify(0), 0);
        assert_eq!(st.classify(1), 1);
        assert_eq!(st.classify(2), 1);
        assert_eq!(st.classify(-1), 2);
        assert_eq!(st.classify(-2), 2);
        assert_eq!(st.classify(3), 3);
        assert_eq!(st.classify(-3), 4);
        // Figure H.2 corners: (0,0) → 0, (0,−L) → 16, (−L,0) → 80,
        // (−L,−L) → 96; X1_Context: small Db → 100, large Db → 129.
        assert_eq!(st.s0(0, 0), 0);
        assert_eq!(st.s0(0, -100), 16);
        assert_eq!(st.s0(-100, 0), 80);
        assert_eq!(st.s0(-100, -100), 96);
        assert_eq!(st.x1(2), 100);
        assert_eq!(st.x1(-2), 100);
        assert_eq!(st.x1(3), 129);
        assert_eq!(st.x1(-3), 129);
        // L = 3 moves the small/zero boundary to 4 (= 2^(L−1), exclusive
        // lower bound) and U = 5 the small/large boundary to 32 (= 2^U,
        // inclusive upper bound).
        let mut st = LosslessStats::new();
        st.l = 3;
        st.u = 5;
        assert_eq!(st.classify(4), 0);
        assert_eq!(st.classify(5), 1);
        assert_eq!(st.classify(32), 1);
        assert_eq!(st.classify(33), 3);
        assert_eq!(st.classify(-33), 4);
    }

    #[test]
    fn byte_source_unstuff() {
        let buf = [0xFF, 0x00, 0x42, 0xFF, 0xD9];
        let mut s = ByteSource::new(&buf);
        assert_eq!(s.next_byte(), 0xFF);
        assert_eq!(s.next_byte(), 0x42);
        assert_eq!(s.next_byte(), 0); // marker triggered
        assert_eq!(s.seen_marker, Some(0xD9));
        assert_eq!(s.next_byte(), 0); // pads
    }
}
