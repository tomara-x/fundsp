//! Dynamical system component.

use super::audionode::*;
use super::combinator::*;
use super::math::*;
use super::signal::*;
use super::*;
use std::marker::PhantomData;

/// A dynamical system is a node that has an attached update function
/// `f(t, dt, x)` where `t` is current time, `dt` is time elapsed since
/// the previous update, and `x` is the enclosed node.
#[derive(Clone)]
pub struct System<T: Float, X: AudioNode, F: FnMut(T, T, &mut X) + Clone> {
    x: X,
    f: F,
    time: T,
    delta_time: T,
    update_interval: T,
    sample_rate: T,
    _marker: PhantomData<T>,
}

impl<T: Float, X: AudioNode, F: FnMut(T, T, &mut X) + Clone> System<T, X, F> {
    /// Create a new dynamical system.
    /// `dt` is the approximate target time between updates.
    pub fn new(x: An<X>, dt: T, f: F) -> Self {
        let mut node = System {
            x: x.0,
            f,
            time: T::zero(),
            delta_time: T::zero(),
            update_interval: dt,
            sample_rate: T::from_f64(DEFAULT_SR),
            _marker: PhantomData::default(),
        };
        let hash = node.ping(true, AttoRand::new(Self::ID));
        node.ping(false, hash);
        node
    }
}

impl<T: Float, X: AudioNode, F: FnMut(T, T, &mut X) + Clone> AudioNode for System<T, X, F> {
    const ID: u64 = 67;
    type Sample = X::Sample;
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    #[inline]
    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.time = T::zero();
        self.delta_time = T::zero();
        if let Some(sr) = sample_rate {
            self.sample_rate = T::from_f64(sr);
        }
    }

    #[inline]
    fn tick(
        &mut self,
        input: &Frame<Self::Sample, Self::Inputs>,
    ) -> Frame<Self::Sample, Self::Outputs> {
        let delta_now = T::new(1) / self.sample_rate;
        // The first update is always done at time zero.
        if self.delta_time >= self.update_interval || self.time == T::zero() {
            (self.f)(self.time, self.delta_time, &mut self.x);
            self.delta_time = T::zero();
        }
        self.time += delta_now;
        self.delta_time += delta_now;
        self.x.tick(input)
    }
    fn process(
        &mut self,
        size: usize,
        input: &[&[Self::Sample]],
        output: &mut [&mut [Self::Sample]],
    ) {
        let delta_now = T::new(size as i64) / self.sample_rate;
        // The first update is always done at time zero.
        if self.delta_time >= self.update_interval || (self.time == T::zero() && size > 0) {
            (self.f)(self.time, self.delta_time, &mut self.x);
            self.delta_time = T::zero();
        }
        self.time += delta_now;
        self.delta_time += delta_now;
        self.x.process(size, input, output);
    }

    #[inline]
    fn ping(&mut self, probe: bool, hash: AttoRand) -> AttoRand {
        self.x.ping(probe, hash.hash(Self::ID))
    }

    fn route(&self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        self.x.route(input, frequency)
    }

    fn set(&mut self, parameter: Tag, value: f64) {
        self.x.set(parameter, value);
    }

    fn get(&self, parameter: Tag) -> Option<f64> {
        self.x.get(parameter)
    }
}
