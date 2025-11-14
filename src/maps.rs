//! workaround for situations where using map isn't possible.
//!
//! don't glob import me!
//! (do `use fundsp::maps;` and call functions as `maps::function()` instead)

use super::audionode::AudioNode;
use super::combinator::An;
use super::hacker::{map, pass, tick, AtomicTable, Interpolation};
use super::math;
use super::Frame;
use core::num::Wrapping;
use numeric_array::typenum::*;
use std::sync::Arc;

pub fn rise() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    (pass() ^ tick()) >> map(|i: &Frame<f32, U2>| f32::from(i[0] > i[1]))
}

pub fn fall() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    (pass() ^ tick()) >> map(|i: &Frame<f32, U2>| f32::from(i[0] < i[1]))
}

pub fn gt() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| f32::from(i[0] > i[1]))
}

pub fn ge() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| f32::from(i[0] >= i[1]))
}

pub fn lt() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| f32::from(i[0] < i[1]))
}

pub fn le() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| f32::from(i[0] <= i[1]))
}

pub fn eq() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| f32::from(i[0] == i[1]))
}

pub fn neq() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| f32::from(i[0] != i[1]))
}

pub fn min() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::min(i[0], i[1]))
}

pub fn max() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::max(i[0], i[1]))
}

pub fn pow() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::pow(i[0], i[1]))
}

pub fn rem_euclid() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| i[0].rem_euclid(i[1]))
}

pub fn rem() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| i[0] % i[1])
}

pub fn rem2() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0] % 2.)
}

pub fn log() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| i[0].log(i[1]))
}

pub fn bitand() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| (i[0] as i32 & i[1] as i32) as f32)
}

pub fn bitor() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| (i[0] as i32 | i[1] as i32) as f32)
}

pub fn bitxor() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| (i[0] as i32 ^ i[1] as i32) as f32)
}

pub fn shl() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| {
        let i = Wrapping(i[0] as i32) << (i[1] as usize);
        i.0 as f32
    })
}

pub fn shr() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| {
        let i = Wrapping(i[0] as i32) >> (i[1] as usize);
        i.0 as f32
    })
}

pub fn bitcrush() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| (i[0] * i[1]).trunc() / i[1])
}

pub fn lerp() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::lerp(i[0], i[1], i[2]))
}

pub fn lerp11() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::lerp11(i[0], i[1], i[2]))
}

pub fn delerp() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::delerp(i[0], i[1], i[2]))
}

pub fn delerp11() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::delerp11(i[0], i[1], i[2]))
}

pub fn xerp() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::xerp(i[0], i[1], i[2]))
}

pub fn xerp11() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::xerp11(i[0], i[1], i[2]))
}

pub fn dexerp() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::dexerp(i[0], i[1], i[2]))
}

pub fn dexerp11() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::dexerp11(i[0], i[1], i[2]))
}

pub fn abs() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::abs(i[0]))
}

pub fn signum() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::signum(i[0]))
}

pub fn floor() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::floor(i[0]))
}

pub fn fract() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].fract())
}

pub fn ceil() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].ceil())
}

pub fn round() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].round())
}

pub fn sqrt() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].sqrt())
}

pub fn exp() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].exp())
}

pub fn exp2() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].exp2())
}

pub fn exp10() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::exp10(i[0]))
}

pub fn exp_m1() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].exp_m1())
}

pub fn ln_1p() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].ln_1p())
}

pub fn ln() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].ln())
}

pub fn log2() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].log2())
}

pub fn log10() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].log10())
}

pub fn hypot() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| i[0].hypot(i[1]))
}

pub fn atan2() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| i[0].atan2(i[1]))
}

pub fn to_pol() -> An<impl AudioNode<Inputs = U2, Outputs = U2>> {
    map(|i: &Frame<f32, U2>| (i[0].hypot(i[1]), i[1].atan2(i[0])))
}

pub fn to_car() -> An<impl AudioNode<Inputs = U2, Outputs = U2>> {
    map(|i: &Frame<f32, U2>| (i[0] * i[1].cos(), i[0] * i[0].sin()))
}

pub fn to_deg() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].to_degrees())
}

pub fn to_rad() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].to_radians())
}

pub fn db_amp() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::db_amp(i[0]))
}

pub fn amp_db() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::amp_db(i[0]))
}

pub fn a_weight() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::a_weight(i[0]))
}

pub fn m_weight() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::m_weight(i[0]))
}

pub fn dissonance() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::dissonance(i[0], i[1]))
}

pub fn dissonance_max() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::dissonance_max(i[0]))
}

pub fn wrap() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::wrap(i[0]))
}

pub fn mirror() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::mirror(i[0]))
}

pub fn recip() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].recip())
}

/// pass input unchanged, but replace nan/inf/subnormals with zeros
pub fn normal() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| if i[0].is_normal() { i[0] } else { 0. })
}

pub fn sin() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].sin())
}

pub fn cos() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].cos())
}

pub fn tan() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].tan())
}

pub fn asin() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].asin())
}

pub fn acos() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].acos())
}

pub fn atan() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].atan())
}

pub fn sinh() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].sinh())
}

pub fn cosh() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].cosh())
}

pub fn tanh() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].tanh())
}

pub fn asinh() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].asinh())
}

pub fn acosh() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].acosh())
}

pub fn atanh() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0].atanh())
}

pub fn squared() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0] * i[0])
}

pub fn cubed() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| i[0] * i[0] * i[0])
}

pub fn spline() -> An<impl AudioNode<Inputs = U5, Outputs = U1>> {
    map(|i: &Frame<f32, U5>| math::spline(i[0], i[1], i[2], i[3], i[4]))
}

pub fn spline_mono() -> An<impl AudioNode<Inputs = U5, Outputs = U1>> {
    map(|i: &Frame<f32, U5>| math::spline_mono(i[0], i[1], i[2], i[3], i[4]))
}

pub fn softsign() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::softsign(i[0]))
}

pub fn softexp() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::softexp(i[0]))
}

pub fn softmix() -> An<impl AudioNode<Inputs = U3, Outputs = U1>> {
    map(|i: &Frame<f32, U3>| math::softmix(i[0], i[1], i[2]))
}

pub fn smooth3() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::smooth3(i[0]))
}

pub fn smooth5() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::smooth5(i[0]))
}

pub fn smooth7() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::smooth7(i[0]))
}

pub fn smooth9() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::smooth9(i[0]))
}

pub fn sine_ease() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::sine_ease(i[0]))
}

pub fn uparc() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::uparc(i[0]))
}

pub fn downarc() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::downarc(i[0]))
}

pub fn sin_hz() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::sin_hz(i[0], i[1]))
}

pub fn cos_hz() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::cos_hz(i[0], i[1]))
}

pub fn sqr_hz() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::sqr_hz(i[0], i[1]))
}

pub fn tri_hz() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::tri_hz(i[0], i[1]))
}

pub fn rnd1() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::rnd1(i[0] as u64) as f32)
}

pub fn rnd2() -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(|i: &Frame<f32, U1>| math::rnd2(i[0] as u64) as f32)
}

pub fn spline_noise() -> An<impl AudioNode<Inputs = U2, Outputs = U1>> {
    map(|i: &Frame<f32, U2>| math::spline_noise(i[0] as u64, i[1]) as f32)
}

pub fn fractal_noise() -> An<impl AudioNode<Inputs = U4, Outputs = U1>> {
    map(|i: &Frame<f32, U4>| math::fractal_noise(i[0] as u64, i[1] as i64, i[2], i[3]) as f32)
}

pub fn atomic_phase(
    t: Arc<AtomicTable>,
    interpolation: Interpolation,
) -> An<impl AudioNode<Inputs = U1, Outputs = U1>> {
    map(move |i: &Frame<f32, U1>| match interpolation {
        Interpolation::Nearest => t.read_nearest(i[0]),
        Interpolation::Linear => t.read_linear(i[0]),
        Interpolation::Cubic => t.read_cubic(i[0]),
    })
}
