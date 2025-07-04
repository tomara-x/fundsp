//! Realtime safe backend for Sequencer.

use super::audiounit::*;
use super::buffer::*;
use super::math::*;
use super::sequencer::*;
use super::signal::*;
#[cfg(feature = "crossbeam")]
use crossbeam_channel::{Receiver, Sender};
#[cfg(not(feature = "crossbeam"))]
use thingbuf::mpsc::{channel, Receiver, Sender};

#[derive(Default, Clone)]
pub(crate) enum Message {
    /// Reset the sequencer.
    #[default]
    Reset,
    /// Add new event in absolute time.
    Push(Event),
    /// Add new event in relative time.
    PushRelative(Event),
    /// Edit event.
    Edit(EventId, Edit),
    /// Edit event in relative time.
    EditRelative(EventId, Edit),
    /// edit loop point
    SetLoop((f64, f64)),
    /// edet replay events flag
    SetReplayEvents(bool),
    /// clear all events
    Clear,
    /// set the sequencer time
    SetTime(f64),
}

#[cfg_attr(feature = "crossbeam", derive(Clone))]
pub struct SequencerBackend {
    /// For sending events for deallocation back to the frontend.
    pub(crate) sender: Sender<Option<Event>>,
    /// For receiving new events from the frontend.
    receiver: Receiver<Message>,
    /// The backend sequencer.
    sequencer: Sequencer,
}

#[cfg(not(feature = "crossbeam"))]
impl Clone for SequencerBackend {
    fn clone(&self) -> Self {
        // Allocate a dummy channel.
        let (sender_1, _receiver_1) = channel(1);
        let (_sender_2, receiver_2) = channel(1);
        SequencerBackend {
            sender: sender_1,
            receiver: receiver_2,
            sequencer: self.sequencer.clone(),
        }
    }
}

impl SequencerBackend {
    /// Create new backend.
    pub(crate) fn new(
        sender: Sender<Option<Event>>,
        receiver: Receiver<Message>,
        sequencer: Sequencer,
    ) -> Self {
        Self {
            sender,
            receiver,
            sequencer,
        }
    }

    /// Handle changes made to the backend.
    fn handle_messages(&mut self) {
        while let Ok(message) = self.receiver.try_recv() {
            match message {
                Message::Reset => {
                    self.reset();
                }
                Message::Push(event) => {
                    self.sequencer.push_event(event);
                }
                Message::PushRelative(event) => {
                    self.sequencer.push_relative_event(event);
                }
                Message::Edit(id, edit) => {
                    self.sequencer.edit(id, edit.end_time, edit.fade_out);
                }
                Message::EditRelative(id, edit) => {
                    self.sequencer
                        .edit_relative(id, edit.end_time, edit.fade_out);
                }
                Message::SetLoop((start, end)) => {
                    self.sequencer.set_loop(start, end);
                }
                Message::SetReplayEvents(keep) => {
                    self.sequencer.set_replay_events(keep);
                }
                Message::Clear => {
                    self.send_back_all();
                    self.sequencer.clear();
                }
                Message::SetTime(t) => {
                    self.sequencer.set_time(t);
                }
            }
        }
    }

    #[inline]
    fn send_back_past(&mut self) {
        while let Some(event) = self.sequencer.get_past_event() {
            if self.sender.try_send(Some(event)).is_ok() {}
        }
    }

    #[inline]
    fn send_back_all(&mut self) {
        while let Some(event) = self.sequencer.get_past_event() {
            if self.sender.try_send(Some(event)).is_ok() {}
        }
        while let Some(event) = self.sequencer.get_ready_event() {
            if self.sender.try_send(Some(event)).is_ok() {}
        }
        while let Some(event) = self.sequencer.get_active_event() {
            if self.sender.try_send(Some(event)).is_ok() {}
        }
    }
}

impl AudioUnit for SequencerBackend {
    fn inputs(&self) -> usize {
        self.sequencer.inputs()
    }

    fn outputs(&self) -> usize {
        self.sequencer.outputs()
    }

    fn reset(&mut self) {
        self.handle_messages();
        if !self.sequencer.replay_events() {
            self.send_back_all();
        }
        self.sequencer.reset();
    }

    fn set_sample_rate(&mut self, sample_rate: f64) {
        self.handle_messages();
        self.sequencer.set_sample_rate(sample_rate);
    }

    #[inline]
    fn tick(&mut self, input: &[f32], output: &mut [f32]) {
        self.handle_messages();
        // Tick and process are the only places where events may be pushed to the past vector.
        if !self.sequencer.replay_events() {
            self.send_back_past();
        }
        if self.sequencer.time() >= self.sequencer.loop_end() {
            self.reset();
        }
        self.sequencer.tick(input, output);
    }

    fn process(&mut self, size: usize, input: &BufferRef, output: &mut BufferMut) {
        self.handle_messages();
        // Tick and process are the only places where events may be pushed to the past vector.
        if !self.sequencer.replay_events() {
            self.send_back_past();
        }
        if self.sequencer.time() >= self.sequencer.loop_end() {
            self.reset();
        }
        self.sequencer.process(size, input, output);
    }

    fn get_id(&self) -> u64 {
        self.sequencer.get_id()
    }

    fn ping(&mut self, probe: bool, hash: AttoHash) -> AttoHash {
        self.handle_messages();
        self.sequencer.ping(probe, hash)
    }

    fn route(&mut self, input: &SignalFrame, frequency: f64) -> SignalFrame {
        self.handle_messages();
        self.sequencer.route(input, frequency)
    }

    fn footprint(&self) -> usize {
        self.sequencer.footprint()
    }

    fn allocate(&mut self) {
        self.sequencer.allocate();
    }
}
