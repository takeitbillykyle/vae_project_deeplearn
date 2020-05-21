#! /usr/bin/env python

#make executable with: chmod +x midiBuffer.py

"""
wrapper for pyPortMidi that handles scheduling so you can send midi events out of order
================================================================================
	midiBuffer.py
	Copyright (C) 2008, Erik Flister (erik.flister@gmail.com)
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
================================================================================
"""

import pygame.midi as pm
import threading
import time
import queue

NOTE_ON=0x90
PRESSURE_CHANGE=0xA0
CONTROL_CHANGE=0xB0
PROGRAM_CHANGE=0xC0
CHANNEL_PRESSURE=0xD0
PITCH_CHANGE=0xE0

NUM_MIDI_CHANS=16
MIDI_MAX=2**7

class PmTimeNegative(Exception):
    "pyPortMidi.Time()<0 -- midiBuffer assumes otherwise."
    pass

class midiBuffer(threading.Thread):
        """
        midiBuffer handles scheduling so you can send midi events out of order.
        it also handles the most common midi output messages for you,
        so you don't have to know how to structure midi packets.
        (quick midi referene: http://www.srm.com/qtma/davidsmidispec.html)
        supported: note on/off w/velocity,
                   pitch bend,
                   key/channel pressures (aka "aftertouch"),
                   controller changes,
                   program changes
        not supported: program bank select,
                       sysex,
                       song/sequencer control,
                       midi time code/timing ticks,
                       any type of input
        requires pyPortMidi 0.0.3 (http://alumni.media.mit.edu/~harrison/pypm.html)
        and python 2.6 (http://www.python.org/download/)
        midiBuffer is a wrapper around pyPortMidi, which is itself a
        wrapper for portmidi, a cross-platform abstraction over native
        midi drivers.  midi drivers require that you send events in
        order.  from portmidi.h:
           Do not expect PortMidi to sort data according to timestamps --
           messages should be sent in the correct order, and timestamps MUST
           be non-decreasing.
        this is especially an issue because the NOTE-OFF message for a
        currently playing note will block any NOTE-ON messages for
        subsequent notes until the current note ends, so you are expected
        to hold on to your NOTE-OFFs until you are sure there will be no
        more intervening NOTE-ONs.  dealing with scheduling in a more
        natural "set-and-forget" way requires concurrency (multithreaded
        programming), which is tricky, and requires tuning to compromise
        between latency and timing inaccuracies.  midiBuffer handles this for you.
        """

        def __init__(self,device=0,appLatency=10,driverLatency=1,verbose=False):
                """
midiBuffer constructor
args:
device          - portmidi device number (set to [] to be prompted)
appLatency      - in ms - how long ahead of schedule midiBuffer should send events to portmidi
                  once they are sent, they will block any events recieved later but scheduled for earlier times
                  so you want low values (also if realtime low-latency response to user input is desired), but
                  if set too low for your processing environment, some events will be sent later than scheduled, resulting in timing jitter
driverLatency   - in ms - portmidi latency (0 means ignore timestamps, so lowest value greater than zero is recommended)
verbose         - print a report useful for tuning appLatency
                """
                self.verbose=verbose
                self.immediately=False
                self.stop=False
                self.incoming=queue.PriorityQueue() #threadsafe
                self.driverLatency=driverLatency
                self.appLatency=appLatency
                self.device=device

                pm.init()

                if self.device==[]:
                        self.opts=0
                        def PrintOutputDevices():
                            print("")
                            print("Choose from the following:")
                            for loop in range(pm.get_count()):
                                interf,name,inp,outp,opened = pm.get_device_info(loop)
                                if outp ==1:
                                    self.opts+=1
                                    if self.opts==1:
                                            self.device=loop
                                    print(loop, name," (", interf, ") ", end="")
                                    if (opened == 1): print("(opened)")
                                    else: print("(unopened)")
                            print("")
                            if self.opts==0:
                                    print("No devices found!")

                        PrintOutputDevices()
                        if self.opts==1:
                                print("Automatically choosing the single output device found")
                        else:
                                self.device = int(input("Type output number: "))
                        print("")

                self.MidiOut=pm.Output(self.device,self.driverLatency)

                threading.Thread.__init__(self)
                self.start()

        def _put(self,item):
                t=pm.time()
                if t<0:
                    if t>-500:
                        time.sleep(abs(t)/1000.0)
                        t=pm.time()

                if t<0:raise PmTimeNegative

                if t>=item[0]-self.appLatency:
                        self.MidiOut.write(item[1])
                else:
                        self.incoming.put(item)

        def run(self):
            t=0

            if self.verbose:
                class report():pass
                rpt=report()
                rpt.sleeps=0
                rpt.latencies=[]

            try:
                while (not self.stop) or (self.incoming!=[] and not(self.incoming.empty())):

                    while pm.time()<=t:
                        if self.verbose: rpt.sleeps+=1
                        time.sleep(.0003) #prevent proc hog - is this too small to actually sleep?  don't want to miss transition to next millisecond...

                    t=pm.time()

                    while (not self.immediately) and self.incoming.queue!=[] and t>=self.incoming.queue[0][0]-self.appLatency: #accessing incoming.queue epends on private implementation of queue.PriorityQueue, hack cuz no peek method provided
                                                                                                    #note a technical concurrency bug here -- queue may become [] after checking for that and before asking for queue[0][0], but since this is the only consumer, shouldn't happen
                            try:
                                    s,e=self.incoming.get_nowait()
                                    if self.verbose: rpt.latencies.append(s-t)
                                    self.MidiOut.write(e)
                            except queue.Empty:
                                    if self.verbose: print("non empty queue throwing empties")
                                    break
            finally:
                if self.immediately:
                    def allNotesOff(m):
                        for i in range(NUM_MIDI_CHANS):
                            for j in range(MIDI_MAX):
                                m.write([[[NOTE_ON + i,j,0],0]])
                    allNotesOff(self.MidiOut)
                if self.verbose and len(rpt.latencies)>0:
                    #print("sleeps: " + str(rpt.sleeps))
                    misses=len(list(filter(lambda x:x<0,rpt.latencies)))
                    def mean(x):
                        return sum(x)/len(x)
                    def std(x):
                        m=mean(x)
                        return mean(list(map(lambda y:abs(y-m),x)))
                    print("latencies (ms, negative=late): mean=" + str(mean(rpt.latencies)) + " std=" + str(std(rpt.latencies)) + " misses=" + str(misses) + " of " + str(len(rpt.latencies)) + " (" + str(100*misses/len(rpt.latencies)) +"%)")
                del self.MidiOut
                pm.quit()

        def close(self,immediately=False):
            """
required clean up -- will block until all events have been sent
args:
immediately - cancels pending events, will send NOTE-OFF to all notes on all channels to prevent hanging notes
            """
            self.immediately=immediately
            self.stop=True
            if self.immediately:
                self.incoming=[]
            elif self.verbose:
                print("waiting for all events to be sent")
            self.join()
            if self.verbose:
                print("midiBuffer done")

        def programChange(self,program,chan=0,onset=0):
                """
send specified program change on a channel at a time
args:
program - program number (0-127)
chan    - midi channel (0-15) (defaults to 0)
onset - in ms, in terms of time returned by midiBuffer.getTime() (defaults to immediately)
                """
                self._put((onset,[[[PROGRAM_CHANGE+chan,program],onset]]))

        def controlChange(self,controller,val=0,chan=0,onset=0):
                """
send specified control change on a channel at a time
args:
controller - controller number (0-127)
val        - control value (0-127) (defaults to 0)
chan       - midi channel (0-15) (defaults to 0)
onset      - in ms, in terms of time returned by midiBuffer.getTime() (defaults to immediately)
                """
                controller = int(controller)
                val = int(val)
                chan = int(chan)
                self._put((onset,[[[CONTROL_CHANGE+chan,controller,int(val)],onset]]))

        def keyPressure(self,note,val=0,chan=0,onset=0):
                """
send specified key pressure change for a note on a channel at a time
args:
note       - key number (0-127)
val        - key pressure (0-127) (defaults to 0)
chan       - midi channel (0-15) (defaults to 0)
onset      - in ms, in terms of time returned by midiBuffer.getTime() (defaults to immediately)
                """
                self._put((onset,[[[PRESSURE_CHANGE+chan,note,int(val)],onset]]))

        def channelPressure(self,val=0,chan=0,onset=0):
                """
send specified channel pressure change on a channel at a time
args:
val        - channel pressure (0-127) (defaults to 0)
chan       - midi channel (0-15) (defaults to 0)
onset      - in ms, in terms of time returned by midiBuffer.getTime() (defaults to immediately)
                """
                self._put((onset,[[[CHANNEL_PRESSURE+chan,int(val)],onset]]))

        def pitchBend(self,val=0,chan=0,onset=0):
                """
send specified pitch bend value on a channel at a time
args:
val        - pitch value (-8192 to +8191, 0 means no bend) (defaults to 0)
chan       - midi channel (0-15) (defaults to 0)
onset      - in ms, in terms of time returned by midiBuffer.getTime() (defaults to immediately)
                """
                val=int(val)
                val+=(MIDI_MAX**2)//2 #center is 8192 (LSB,MSB = 0x00,0x40)
                MSB=val//MIDI_MAX
                LSB=val-MSB*MIDI_MAX
                self._put((onset,[[[PITCH_CHANGE+chan,LSB,MSB],onset]]))

        def playChord(self,notes,dur,vel=MIDI_MAX//2,chan=0,onset=0,NoteOffMode="schedule"):
            """
play a list of notes at the specified midi channel, velocity, duration, and onset time (NOTE-OFF handled for you automatically)
args:
notes - list of midi note numbers (0-127)
dur   - in ms
vel   - midi velocity (0-127) (defaults to 64)
chan  - midi channel (0-15) (defaults to 0)
onset - in ms, in terms of time returned by midiBuffer.getTime() (defaults to immediately)
NoteOffMode - if "schedule" (default), schedule a note-off after dur. if "hack", play a note-off immediately before on the same pitch INSTEAD.
            """
            ChordList = []
            for i in range(len(notes)):
                if NoteOffMode == "hack":
                    ChordList.append([[NOTE_ON + chan,notes[i],0], onset-0.01])
                ChordList.append([[NOTE_ON + chan,notes[i],vel], onset  ])
            self._put((onset,ChordList))
            # if NoteOffMode == "hack":
            #     return
            if onset==0:
                    onset=self.getTime()
            ChordList = []
            offset=onset+dur
            for i in range(len(notes)):
                ChordList.append([[NOTE_ON + chan,notes[i],0],offset]) #velocity=0 is equivalent to NOTE_OFF
            self._put((offset,ChordList))


        def getTime(self):
            """
get current time in ms, use to generate onset timestamps to send to midiBuffer.playChord()
            """
            return pm.time()

def test():
    b = midiBuffer(device=[], verbose=True)
    pitch = 60
    b.playChord([pitch], dur=3000, onset=0  , NoteOffMode="hack")
    b.playChord([pitch], dur=3000, onset=300, NoteOffMode="hack")
    b.playChord([pitch], dur=3000, onset=600, NoteOffMode="hack")
    b.playChord([pitch], dur=3000, onset=900, NoteOffMode="hack")
    # import random
    # for i in range(10):
    #     pitch = random.choice(range(40, 80))
    #     b.playChord([pitch], dur=500.0, onset=i*200)
    time.sleep(5)
    b.close()

def test_multichannel():
    b = midiBuffer(device=[], verbose=True)
    t = b.getTime()
    pitch = 60
    for i in range(5):
        for j in range(3):
            b.playChord([40 + i*2 + j*3], dur=100, onset=t*100, chan=j)
            t += 1
    # import random
    # for i in range(10):
    #     pitch = random.choice(range(40, 80))
    #     b.playChord([pitch], dur=500.0, onset=i*200)
    time.sleep(5)
    b.close()
    
if __name__ == "__main__":

        #test()
        test_multichannel()
        quit()
        
        b=midiBuffer(device=[],verbose=True)
        tempo=160
        dur=int(60.0/tempo/2 * 1000) #(eigth note duration in ms)
        notes=[60, 63, 66, 70]
        pitches=[]
        maxPitch=MIDI_MAX**2
        for i in range(dur):
                pitches.append(int((i*(maxPitch-1.0)/(dur-1)) - maxPitch/2))

        n=0
        t=b.getTime()

        for n in range(10):
                #b.programChange(n,1,t)
                #b.playChord(notes,dur,127,1,t) #without midiBuffer, the NOTE-OFF's from this chord would blcok the following events
                #b.playChord(notes,dur,127,onset=t)

                for i in range(dur):
                        b.playChord(notes, dur, 127, onset=t+i)
                        b.controlChange(1,int(i*127.0/(dur-1)),1,t+i) #controller 1 is the mod wheel
                        b.pitchBend(pitches[i],1,t+i)
                        b.keyPressure(notes[0]-12,int(i*127.0/(dur-1)),1,t+i)
                        b.channelPressure(int(i*127.0/(dur-1)),1,t+i)

                t+=2*dur

        b.close()