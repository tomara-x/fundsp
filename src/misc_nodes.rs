//! various nodes/units (some are WIP)

use super::audionode::*;
use super::audiounit::*;
use super::buffer::*;
use super::fft::*;
use super::math::*;
use super::signal::*;
use super::*;
use numeric_array::typenum::*;

extern crate alloc;
use alloc::sync::Arc;

#[cfg(feature = "crossbeam")]
pub use buff_nodes::*;

#[cfg(feature = "crossbeam")]
mod buff_nodes {
    use super::*;
    use crossbeam_channel::{Receiver, Sender};
    /// send samples to crossbeam channel
    /// use this and [`BuffOut`] with a crossbeam channel to make an audio portal
    /// that you can put either end of in any part of your graph
    /// (channel must have capacity larger than the audio stream buffer size)
    /// ```
    /// let (s, r) = crossbeam_channel::bounded(1);
    /// let i = An(BuffIn::new(s));
    /// let o = An(BuffOut::new(r));
    /// let mut feedback_graph = (o+pass()) >> i;
    /// println!("{:?}", feedback_graph.tick(&Frame::from([1.])));
    /// println!("{:?}", feedback_graph.tick(&Frame::from([1.])));
    /// ```
    ///
    /// - input 0: input
    /// - output 0: input passed through
    #[derive(Clone)]
    pub struct BuffIn {
        s: Sender<f32>,
    }
    impl BuffIn {
        pub fn new(s: Sender<f32>) -> Self {
            BuffIn { s }
        }
    }
    impl AudioNode for BuffIn {
        const ID: u64 = 1123;
        type Inputs = U1;
        type Outputs = U1;

        #[inline]
        fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
            let _ = self.s.try_send(input[0]);
            [input[0]].into()
        }
    }

    /// receive samples from crossbeam channel
    /// - output 0: output
    #[derive(Clone)]
    pub struct BuffOut {
        r: Receiver<f32>,
    }
    impl BuffOut {
        pub fn new(r: Receiver<f32>) -> Self {
            BuffOut { r }
        }
    }
    impl AudioNode for BuffOut {
        const ID: u64 = 1124;
        type Inputs = U0;
        type Outputs = U1;

        #[inline]
        fn tick(&mut self, _input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
            [self.r.try_recv().unwrap_or_default()].into()
        }
    }
}

// rfft and ifft accumulate input, they have a combined latency of 2 window lengths
/// rfft
/// - input 0: input
/// - output 0: real
/// - output 1: imaginary
#[derive(Default, Clone)]
pub struct Rfft {
    n: usize,
    data: Vec<f32>,
    count: usize,
    start: usize,
}
impl Rfft {
    pub fn new(n: usize, offset: usize) -> Self {
        let n = n.clamp(2, 32768).next_power_of_two();
        let start = (n - offset) % n;
        let data = vec![0.; n * 2];
        Rfft {
            n,
            data,
            count: start,
            start,
        }
    }
}
impl AudioNode for Rfft {
    const ID: u64 = 1120;
    type Inputs = U1;
    type Outputs = U2;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let i = self.count;
        self.count += 1;
        if self.count == self.n {
            self.count = 0;
        }
        if i == 0 {
            real_fft(&mut self.data[..self.n]);
            fix_nyquist(&mut self.data[..self.n + 2]);
            // fix negative frequencies
            let mut i = self.n + 2;
            let len = self.n * 2;
            while i < len {
                self.data[i] = self.data[len - i];
                self.data[i + 1] = -self.data[len - i + 1];
                i += 2;
            }
        }
        let j = i * 2;
        let out = [self.data[j], self.data[j + 1]];
        self.data[i] = input[0];
        out.into()
    }

    fn reset(&mut self) {
        self.count = self.start;
        self.data.fill(0.);
    }
}

/// ifft
/// - input 0: real
/// - input 1: imaginary
/// - output 0: real
/// - output 1: imaginary
#[derive(Default, Clone)]
pub struct Ifft {
    n: usize,
    data: Vec<Complex32>,
    count: usize,
    start: usize,
}
impl Ifft {
    pub fn new(n: usize, offset: usize) -> Self {
        let n = n.clamp(2, 32768).next_power_of_two();
        let start = (n - offset) % n;
        let data = vec![Complex32::ZERO; n];
        Ifft {
            n,
            data,
            count: start,
            start,
        }
    }
}
impl AudioNode for Ifft {
    const ID: u64 = 1121;
    type Inputs = U2;
    type Outputs = U2;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let i = self.count;
        self.count += 1;
        if self.count == self.n {
            self.count = 0;
        }
        if i == 0 {
            inverse_fft(&mut self.data);
        }
        let out = [self.data[i].re, self.data[i].im];
        self.data[i] = Complex32::new(input[0], input[1]);
        out.into()
    }

    fn reset(&mut self) {
        self.count = self.start;
        self.data.fill(Complex32::ZERO);
    }
}

/// switch between units based on index.
/// units must have 0 inputs, and 1 output
/// - input 0: index
/// - output 0: output from selected unit
#[derive(Default, Clone)]
pub struct Select {
    units: Vec<Box<dyn AudioUnit>>,
}

impl Select {
    pub fn new(units: Vec<Box<dyn AudioUnit>>) -> Self {
        Select { units }
    }
}

impl AudioNode for Select {
    const ID: u64 = 1213;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        if let Some(unit) = self.units.get_mut(input[0] as usize) {
            unit.tick(&[], &mut buffer);
        }
        buffer.into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        for unit in &mut self.units {
            unit.set_sample_rate(sample_rate);
        }
    }

    fn reset(&mut self) {
        for unit in &mut self.units {
            unit.reset();
        }
    }
}

/// fading switch between units based on index.
/// (interpolate on fractional index).
/// units must have 0 inputs, and 1 output
/// - input 0: index
/// - output 0: output from selected unit
#[derive(Default, Clone)]
pub struct FadeSelect {
    units: Vec<Box<dyn AudioUnit>>,
}

impl FadeSelect {
    pub fn new(units: Vec<Box<dyn AudioUnit>>) -> Self {
        FadeSelect { units }
    }
}

impl AudioNode for FadeSelect {
    const ID: u64 = 61636162;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        let index = input[0] as usize;
        if let Some(unit) = self.units.get_mut(index) {
            unit.tick(&[], &mut buffer);
            let fract = input[0].fract();
            if fract != 0. {
                if let Some(unit) = self.units.get_mut(index + 1) {
                    let mut tmp = [0.];
                    unit.tick(&[], &mut tmp);
                    buffer[0] = lerp(buffer[0], tmp[0], fract);
                }
            }
        }
        buffer.into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        for unit in &mut self.units {
            unit.set_sample_rate(sample_rate);
        }
    }

    fn reset(&mut self) {
        for unit in &mut self.units {
            unit.reset();
        }
    }
}

/// sequence units.
/// units must have 0 inputs, and 1 output
/// - input 0: trigger
/// - input 1: index of unit to play
/// - input 2: delay time
/// - input 3: duration
/// - output 0: output from all playing units
#[derive(Default, Clone)]
pub struct Seq {
    units: Vec<Box<dyn AudioUnit>>,
    // index, delay, duration (times in samples)
    events: Vec<(usize, usize, usize)>,
    sr: f32,
}

impl Seq {
    pub fn new(units: Vec<Box<dyn AudioUnit>>) -> Self {
        Seq {
            units,
            events: Vec::with_capacity(1024),
            sr: 44100.,
        }
    }
}

impl AudioNode for Seq {
    const ID: u64 = 1729;
    type Inputs = U4;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        // triggered, add an event
        if input[0] != 0. {
            // remove existing events for that index
            self.events.retain(|&x| x.0 != input[1] as usize);
            // reset the unit
            if let Some(unit) = self.units.get_mut(input[1] as usize) {
                unit.reset();
            }
            // push the new event
            self.events.push((
                input[1] as usize,
                (input[2] * self.sr).round() as usize,
                (input[3] * self.sr).round() as usize,
            ));
        }
        // remove finished events
        self.events.retain(|&x| x.2 != 0);

        let mut buffer = [0.];
        let mut out = [0.];
        for i in &mut self.events {
            if i.1 == 0 {
                if let Some(unit) = self.units.get_mut(i.0) {
                    unit.tick(&[], &mut buffer);
                    out[0] += buffer[0];
                }
                i.2 -= 1;
            } else {
                i.1 -= 1;
            }
        }
        out.into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sr = sample_rate as f32;
        for unit in &mut self.units {
            unit.set_sample_rate(sample_rate);
        }
    }

    fn reset(&mut self) {
        for unit in &mut self.units {
            unit.reset();
        }
    }
}

/// index an array of floats.
/// - input 0: index
/// - output 0: value at index
#[derive(Clone)]
pub struct ArrGet {
    arr: Arc<Vec<f32>>,
}

impl ArrGet {
    pub fn new(arr: Arc<Vec<f32>>) -> Self {
        ArrGet { arr }
    }
}

impl AudioNode for ArrGet {
    const ID: u64 = 1312;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        if let Some(n) = self.arr.get(input[0] as usize) {
            buffer[0] = *n;
        }
        buffer.into()
    }
}

// TODO(amy): s&h nodes in a net instead?
/// shift register.
/// - input 0: input signal
/// - input 1: trigger
/// - output 0...8: output from each index
#[derive(Default, Clone)]
pub struct ShiftReg {
    reg: [f32; 8],
}

impl ShiftReg {
    pub fn new() -> Self {
        ShiftReg { reg: [0.; 8] }
    }
}

impl AudioNode for ShiftReg {
    const ID: u64 = 1110;
    type Inputs = U2;
    type Outputs = U8;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        if input[1] != 0. {
            self.reg[7] = self.reg[6];
            self.reg[6] = self.reg[5];
            self.reg[5] = self.reg[4];
            self.reg[4] = self.reg[3];
            self.reg[3] = self.reg[2];
            self.reg[2] = self.reg[1];
            self.reg[1] = self.reg[0];
            self.reg[0] = input[0];
        }
        self.reg.into()
    }

    fn reset(&mut self) {
        self.reg = [0., 0., 0., 0., 0., 0., 0., 0.];
    }
}

/// quantizer.
/// this assumes all steps are non-negative, ordered, and contain range limits
/// example:
/// ```
/// use fundsp::hacker32::*;
/// // quantize to natural minor (in semis)
/// let scale = vec![0., 2., 3., 5., 7., 8., 10., 12.];
/// let range = 12.;
/// let mut q = An(Quantizer::new(scale, range));
/// assert_eq!(q.filter_mono(4.), 3.);
/// assert_eq!(q.filter_mono(14.), 14.);
/// ```
/// - input 0: value to quantize
/// - output 0: quantized value
#[derive(Clone)]
pub struct Quantizer {
    arr: Vec<f32>,
    range: f32,
}

impl Quantizer {
    pub fn new(arr: Vec<f32>, range: f32) -> Self {
        Quantizer { arr, range }
    }
}

impl AudioNode for Quantizer {
    const ID: u64 = 1111;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        let n = input[0];
        let wrapped = n - self.range * (n / self.range).floor();
        let mut nearest = 0.;
        let mut dist = f32::MAX;
        for i in &self.arr {
            let d = (wrapped - i).abs();
            if d < dist {
                nearest = *i;
                dist = d;
            }
        }
        buffer[0] = n + nearest - wrapped;
        buffer.into()
    }
}

/// tick a unit every n samples.
/// - inputs 0..: inputs to the unit
/// - outputs 0..: last outputs from the unit
#[derive(Clone)]
pub struct Kr {
    x: Box<dyn AudioUnit>,
    n: usize,
    out_buffer: Vec<f32>,
    in_buffer: Vec<f32>,
    count: usize,
    inputs: usize,
    outputs: usize,
    // set the sr of the inner unit to sr/n to keep durations and frequencies unchanged
    preserve_time: bool,
}

impl Kr {
    pub fn new(x: Box<dyn AudioUnit>, n: usize, preserve_time: bool) -> Self {
        let inputs = x.inputs();
        let outputs = x.outputs();
        let out_buffer = vec![0.; outputs];
        let in_buffer = vec![0.; inputs];
        let n = std::cmp::Ord::max(n, 1);
        Kr {
            x,
            n,
            out_buffer,
            in_buffer,
            count: 0,
            inputs,
            outputs,
            preserve_time,
        }
    }
}

impl AudioUnit for Kr {
    fn reset(&mut self) {
        self.x.reset();
        self.count = 0;
        self.out_buffer.fill(0.);
        self.in_buffer.fill(0.);
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        if self.preserve_time {
            self.x.set_sample_rate(sample_rate / self.n as f64);
        } else {
            self.x.set_sample_rate(sample_rate);
        }
    }

    fn tick(&mut self, input: &[f32], output: &mut [f32]) {
        if self.count == 0 {
            self.count = self.n;
            self.x.tick(input, &mut self.out_buffer);
        }
        self.count -= 1;
        output.copy_from_slice(&self.out_buffer);
    }

    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        let mut i = 0;
        while i < size {
            if self.count == 0 {
                self.count = self.n;
                for c in 0..input.channels() {
                    self.in_buffer[c] = input.at_f32(c, i);
                }
                self.x.tick(&self.in_buffer, &mut self.out_buffer);
            }
            self.count -= 1;
            for c in 0..output.channels() {
                output.set_f32(c, i, self.out_buffer[c]);
            }
            i += 1;
        }
    }

    fn inputs(&self) -> usize {
        self.inputs
    }

    fn outputs(&self) -> usize {
        self.outputs
    }

    fn route(&mut self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Arbitrary(0.0).route(input, self.outputs())
    }

    fn get_id(&self) -> u64 {
        const ID: u64 = 1112;
        ID
    }

    fn ping(&mut self, probe: bool, hash: AttoHash) -> AttoHash {
        self.x.ping(probe, hash.hash(self.get_id()))
    }

    fn footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }

    fn allocate(&mut self) {
        self.x.allocate();
    }
}

/// reset unit every s seconds.
/// unit must have 0 inputs and 1 output
/// - output 0: output from the unit
#[derive(Clone)]
pub struct Reset {
    unit: Box<dyn AudioUnit>,
    dur: f32,
    n: usize,
    count: usize,
}

impl Reset {
    pub fn new(unit: Box<dyn AudioUnit>, s: f32) -> Self {
        Reset {
            unit,
            dur: s,
            n: (s * 44100.).round() as usize,
            count: 0,
        }
    }
}

impl AudioNode for Reset {
    const ID: u64 = 1113;
    type Inputs = U0;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, _input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        if self.count >= self.n {
            self.unit.reset();
            self.count = 0;
        }
        self.unit.tick(&[], &mut buffer);
        self.count += 1;
        buffer.into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.n = (self.dur * sample_rate as f32).round() as usize;
        self.unit.set_sample_rate(sample_rate);
    }

    fn reset(&mut self) {
        self.count = 0;
        self.unit.reset();
    }
}

/// reset unit when triggered.
/// unit must have 0 inputs and 1 output
/// - input 0: reset the unit when non-zero
/// - output 0: output from the unit
#[derive(Clone)]
pub struct TrigReset {
    unit: Box<dyn AudioUnit>,
}

impl TrigReset {
    pub fn new(unit: Box<dyn AudioUnit>) -> Self {
        TrigReset { unit }
    }
}

impl AudioNode for TrigReset {
    const ID: u64 = 1114;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        if input[0] != 0. {
            self.unit.reset();
        }
        self.unit.tick(&[], &mut buffer);
        buffer.into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.unit.set_sample_rate(sample_rate);
    }

    fn reset(&mut self) {
        self.unit.reset();
    }
}

/// reset unit every s seconds (duration as input).
/// unit must have 0 inputs and 1 output
/// - input 0: reset interval
/// - output 0: output from the unit
#[derive(Clone)]
pub struct ResetV {
    unit: Box<dyn AudioUnit>,
    count: usize,
    sr: f32,
}

impl ResetV {
    pub fn new(unit: Box<dyn AudioUnit>) -> Self {
        ResetV {
            unit,
            count: 0,
            sr: 44100.,
        }
    }
}

impl AudioNode for ResetV {
    const ID: u64 = 1115;
    type Inputs = U1;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut buffer = [0.];
        if self.count >= (input[0] * self.sr).round() as usize {
            self.unit.reset();
            self.count = 0;
        }
        self.unit.tick(&[], &mut buffer);
        self.count += 1;
        buffer.into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sr = sample_rate as f32;
        self.unit.set_sample_rate(sample_rate);
    }

    fn reset(&mut self) {
        self.count = 0;
        self.unit.reset();
    }
}

/// sample and hold.
/// - input 0: input signal
/// - input 1: trigger
/// - output 0: held signal
#[derive(Clone)]
pub struct SnH {
    val: f32,
}

impl Default for SnH {
    fn default() -> Self {
        Self::new()
    }
}

impl SnH {
    pub fn new() -> Self {
        SnH { val: 0. }
    }
}

impl AudioNode for SnH {
    const ID: u64 = 1125;
    type Inputs = U2;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        if input[1] != 0. {
            self.val = input[0];
        }
        [self.val].into()
    }

    fn reset(&mut self) {
        self.val = 0.;
    }
}

/// AudioUnit version of [`MultiSplit`]
#[derive(Clone)]
pub struct MultiSplitUnit {
    inputs: usize,
    outputs: usize,
}
impl MultiSplitUnit {
    pub fn new(inputs: usize, splits: usize) -> Self {
        let outputs = inputs * splits;
        MultiSplitUnit { inputs, outputs }
    }
}
impl AudioUnit for MultiSplitUnit {
    fn reset(&mut self) {}

    fn set_sample_rate(&mut self, _sample_rate: f64) {}

    fn tick(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.outputs {
            output[i] = input[i % self.inputs];
        }
    }

    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        for channel in 0..self.outputs {
            for i in 0..simd_items(size) {
                output.set(channel, i, input.at(channel % self.inputs, i));
            }
        }
    }

    fn inputs(&self) -> usize {
        self.inputs
    }

    fn outputs(&self) -> usize {
        self.outputs
    }

    fn route(&mut self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Split.route(input, self.outputs())
    }

    fn get_id(&self) -> u64 {
        const ID: u64 = 138;
        ID
    }

    fn footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

/// AudioUnit version of [`MultiJoin`]
#[derive(Clone)]
pub struct MultiJoinUnit {
    outputs: usize,
    branches: usize,
}
impl MultiJoinUnit {
    pub fn new(outputs: usize, branches: usize) -> Self {
        MultiJoinUnit { outputs, branches }
    }
}
impl AudioUnit for MultiJoinUnit {
    fn reset(&mut self) {}

    fn set_sample_rate(&mut self, _sample_rate: f64) {}

    fn tick(&mut self, input: &[f32], output: &mut [f32]) {
        for j in 0..self.outputs {
            let mut out = input[j];
            for i in 1..self.branches {
                out += input[j + i * self.outputs];
            }
            output[j] = out / self.branches as f32;
        }
    }

    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        let z = 1.0 / self.branches as f32;
        for channel in 0..self.outputs {
            for i in 0..simd_items(size) {
                output.set(channel, i, input.at(channel, i) * z);
            }
        }
        for channel in self.outputs..self.outputs * self.branches {
            for i in 0..simd_items(size) {
                output.add(channel % self.outputs, i, input.at(channel, i) * z);
            }
        }
    }

    fn inputs(&self) -> usize {
        self.outputs * self.branches
    }

    fn outputs(&self) -> usize {
        self.outputs
    }

    fn route(&mut self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Join.route(input, self.outputs())
    }

    fn get_id(&self) -> u64 {
        const ID: u64 = 139;
        ID
    }

    fn footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

/// AudioUnit version of [`Reverse`]
#[derive(Clone)]
pub struct ReverseUnit {
    n: usize,
}
impl ReverseUnit {
    pub fn new(n: usize) -> Self {
        ReverseUnit { n }
    }
}
impl AudioUnit for ReverseUnit {
    fn reset(&mut self) {}

    fn set_sample_rate(&mut self, _sample_rate: f64) {}

    fn tick(&mut self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.n {
            output[i] = input[self.n - 1 - i];
        }
    }

    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        for channel in 0..self.n {
            for i in 0..simd_items(size) {
                output.set(channel, i, input.at(self.n - 1 - channel, i));
            }
        }
    }

    fn inputs(&self) -> usize {
        self.n
    }

    fn outputs(&self) -> usize {
        self.n
    }

    fn route(&mut self, input: &SignalFrame, _frequency: f64) -> SignalFrame {
        Routing::Reverse.route(input, self.n)
    }

    fn get_id(&self) -> u64 {
        const ID: u64 = 145;
        ID
    }

    fn footprint(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

/// euclidean rhythm generator.
/// stolen (with modifications) from:
/// [github.com/grindcode/rhythms](https://github.com/grindcode/rhythms)
/// - input 0: trigger (step forward when non-zero)
/// - input 1: length
/// - input 2: pulses
/// - input 3: rotation
/// - output 0: euclidean sequence
#[derive(Clone)]
pub struct EuclidSeq {
    steps: Vec<f32>,
    length: usize,
    pulses: usize,
    rotation: isize,
    cursor: usize,
}

impl Default for EuclidSeq {
    fn default() -> Self {
        Self::new()
    }
}

impl EuclidSeq {
    pub fn new() -> Self {
        EuclidSeq {
            steps: Vec::with_capacity(128),
            length: 0,
            pulses: 0,
            rotation: 0,
            cursor: 0,
        }
    }
}

impl AudioNode for EuclidSeq {
    const ID: u64 = 41434142;
    type Inputs = U4;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let mut out = [0.];
        if input[0] != 0. {
            let l = input[1] as usize;
            let l = l.clamp(1, 128);
            let p = input[2] as usize;
            let p = std::cmp::Ord::min(l, p);
            let r = input[3] as isize;
            if l != self.length || p != self.pulses {
                self.pulses = p;
                self.length = l;
                self.steps.clear();
                let mut bucket: usize = 0;
                for _ in 0..self.length {
                    bucket += self.pulses;
                    if bucket >= self.length {
                        bucket -= self.length;
                        self.steps.push(1.);
                    } else {
                        self.steps.push(0.);
                    }
                }
                if self.length > 0 && self.pulses > 0 {
                    let offset = self.length / self.pulses - 1;
                    self.steps.rotate_right(offset);
                }
                if self.cursor >= self.length {
                    self.cursor = 0;
                }
            }
            if r != self.rotation {
                self.rotation = r;
                if r.is_positive() {
                    self.steps.rotate_right(r as usize);
                } else if r.is_negative() {
                    self.steps.rotate_left(r.unsigned_abs());
                }
            }
            out[0] = self.steps[self.cursor];
            self.cursor += 1;
            if self.cursor >= self.length {
                self.cursor = 0;
            }
        }
        out.into()
    }

    fn set_sample_rate(&mut self, _sample_rate: f64) {}

    fn reset(&mut self) {
        self.steps.clear();
        self.length = 0;
        self.pulses = 0;
        self.rotation = 0;
        self.cursor = 0;
    }
}

/// output 1 for the given duration, 0 after.
/// - output 0: gate signal
#[derive(Clone)]
pub struct Gate {
    dur: f64,
    t: f64,
    sample_duration: f64,
    output: f32,
}

impl Gate {
    pub fn new(dur: f64) -> Self {
        let t = 0.;
        let sample_duration = 1. / DEFAULT_SR;
        let output = 1.;
        Gate {
            dur,
            t,
            sample_duration,
            output,
        }
    }
}

impl AudioNode for Gate {
    const ID: u64 = 520;
    type Inputs = U0;
    type Outputs = U1;

    #[inline]
    fn tick(&mut self, _input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        if self.t >= self.dur {
            self.output = 0.;
        }
        self.t += self.sample_duration;
        [self.output].into()
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_duration = 1. / sample_rate;
    }

    fn reset(&mut self) {
        self.t = 0.;
        self.output = 1.;
    }
}
