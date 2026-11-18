# Coding guide: narrative labels for police-recorded e-scooter crashes

**Version 1.2, 2026-09-14.** Supplementary S3.
Written by the study author from 50 narratives drawn at random. Those 50 narratives are excluded from everything you will code.

---

> *The worked examples in this copy are constructed. The guide the coders used quoted crash narratives here; each quotation is replaced by an example that makes the same point and keeps the same label, because the narrative text is not redistributed.*

## 1. What you are doing

You will read police crash narratives one at a time and mark fifteen yes/no labels for each. Ten labels describe the **circumstances before the crash**. Five describe **what the narrative says about injury**.

You are recording **what the narrative states**, not what you believe happened and not who was at fault. If the narrative does not say it, the answer is no.

Three people are coding the same narratives independently. Do not discuss any narrative with the other coders. There is no answer key and no discussion step afterward; disagreement between coders is a measurement we report, not a mistake to be corrected.

You will see no output from any automated system. Nothing you code is checked against a computer's answer before you finish.

Work in sittings of about an hour. Export your CSV at the end of every sitting.

---

## 2. The one rule that comes before all the others: find the scooter

**The scooter is not always "Unit 2."** Reports number the units inconsistently, and some reports renumber them partway through.

Before you code anything, read the narrative once and decide **which unit is the electric scooter**. Identify it by description, never by number:

- "electric scooter", "e-scooter", "motorized scooter", "motor assisted scooter", "stand up scooter", "motorized conveyance", "motorized LIME scooter"
- brand names: Bird, Lime, Gotrax, Jetta, Spin, Veo
- a phrase such as "a pedestrian on a motorized scooter" (some reports code the rider as a pedestrian; that unit is still the scooter)

Units appear as `Unit 1`, `Unit #2`, `UNIT1`, `U1`, `Person 2`, `Vehicle 1`. All of these are unit labels.

In this study, every crash involves one electric scooter and one motor vehicle. The **rider** is the person on the scooter. The **driver** is the person in the motor vehicle.

**If the report renumbers the units mid-narrative** (for example, "UNIT 1 WILL BECOME UNIT #2"), code from the corrected assignment the officer settled on, and tick the "not sure" box on any label the renumbering affects.

**If you genuinely cannot tell which unit is the scooter**, code only what is unambiguous and tick "not sure" on everything else.

---

## 3. How to mark a label

- **1 (yes)** — the narrative states this, or describes it in plain words.
- **0 (no)** — the narrative does not state it, states the opposite, or is silent.

Silence is 0. A narrative that says nothing about the sidewalk gets 0 for sidewalk transition. You are not guessing what probably happened.

**"Not sure" flag.** Tick it when you had to make a close call. The flag is recorded separately and does not change your 0 or 1; mark the label with your best reading and flag it. Use it freely, it costs nothing.

**Notes field.** One line on anything odd. Especially useful when the narrative contradicts itself.

**More than one label can be 1.** The ten circumstance labels are not exclusive. A narrative can describe a rider leaving a sidewalk, failing to yield, and a vehicle turning across, all at once. Mark every one that applies.

---

## 4. Fault is not your business

Police narratives assign fault, cite statutes, and say who was placed at fault. **Ignore all of it.** You are coding described actions, not findings.

A narrative that says "Unit 1 was placed at fault for failure to yield", and nothing more, gets 0 for failure to yield: a fault finding is not a description of what happened.

The reverse also holds. Where the narrative describes an action, mark the label even if the officer blamed the other party. If the narrative says the rider ran a stop sign but placed the driver at fault, signal violation is still 1.

---

## 5. The ten circumstance labels

For each label: what it is, what counts, what does not, real examples taken from narratives you will not be coding, and the rule for the hard case.

---

### 5.1 Wrong-way or contraflow travel

**What it is.** The scooter was traveling against the legal direction of traffic for the space it was in.

**Counts:**
- riding against traffic, the wrong way, or in the opposite direction of travel
- riding in a bike lane or shoulder on the wrong side of the road for that direction

**Does not count:**
- riding on a sidewalk (that is 5.2, not this)
- crossing a road from one side to the other (crossing is not contraflow)
- the motor vehicle traveling the wrong way

**Examples.**
> "Unit 2, an e-scooter, rode north in the bike lane on the side of the street meant for southbound traffic." → **1**

> "Unit 2, an e-scooter, rode in the right travel lane because the street had no sidewalk." → **0** (lane position, direction not contrary)

**Hard case.** If the narrative gives directions that imply contraflow but never says so, and you had to reconstruct it from compass headings, code 1 only if the narrative states both the scooter's direction and the direction of the lane it was in. Otherwise 0 and flag.

---

### 5.2 Sidewalk to roadway transition

**What it is.** The scooter moved from a sidewalk, path, or off-roadway area into the roadway, or was crossing the roadway having come from a sidewalk.

**Counts:**
- riding on the sidewalk and then entering the street, a crosswalk, or the travel lanes
- crossing a roadway from a sidewalk, **whether or not a marked crosswalk is used**
- entering the roadway from a shoulder, path, curb, or trail

**Does not count:**
- riding along a sidewalk with no entry into the roadway described
- entering the roadway from a driveway, alley, or parking lot (that is 5.3)
- the crash occurring on the sidewalk itself with no roadway entry

**Examples.**
> "Unit 2 rode an e-scooter east along the sidewalk, then went off the curb to cross the street." → **1**

> "Unit 2 rode an e-scooter west along the sidewalk, could not slow down and hit Unit 1 as it backed out of a driveway." → **0** for this label (the scooter stayed on the sidewalk; the roadway was not entered)

**Hard case: crosswalks.** This is the most common pattern in these data and the rule is fixed here so that all three of you apply the same one.

- **Crossing a roadway in a marked crosswalk, having come from a sidewalk or curb → code 1.** The rider left an off-roadway space and entered the roadway.
- **Crossing a roadway outside a crosswalk, from a sidewalk or curb → code 1.** Same movement, no marking.
- **Crossing where the narrative never says where the rider came from → code 0 and flag.** "Unit 2 was crossing on the crosswalk" with no prior position stated is not enough.
- **Riding along inside a crosswalk parallel to traffic, or stopped in one → code 0.**

> "Unit 2 crossed the street in the crosswalk from the west sidewalk and was hit as it reached the east sidewalk." → **1** (crossing the roadway between sidewalks)

> "Unit 2 rode south through the crosswalk over the eastbound lanes." → **0 and flag** (crossing, but the narrative never says the rider came from a sidewalk or curb)

---

### 5.3 Entry from a driveway or alley

**What it is.** Either unit entered or left the roadway through a driveway, alley, private drive, or parking-lot exit.

**Counts:**
- the scooter entering the road from a driveway, alley, private drive, or parking lot
- the motor vehicle backing out of, turning into, or exiting a driveway or parking lot
- "private drive", "driveway", "alley", "parking lot entrance/exit"

**Does not count:**
- a crash that happens entirely inside a parking lot with no roadway entry
- a street intersection, however small

**Examples.**
> "Unit 2 pulled out of a business parking lot and did not yield as it left the private drive." → **1**

> "Unit 1 was reversing out of a residential driveway." → **1**

**Hard case.** A crash wholly inside a parking lot (neither unit entering or leaving the roadway) is **0**. Flag it.

---

### 5.4 Failure to yield the right of way

**What it is.** The narrative uses yield or right-of-way language about one of the parties.

This label is narrower than it sounds, and deliberately so. **You are recording that the officer wrote that someone failed to yield.** You are not being asked to judge whether someone should have yielded.

**Mark 1 only when the narrative contains yield or right-of-way language**, such as:
- "failed to yield", "failing to yield", "fails to yield"
- "did not yield", "without yielding"
- "yield the right of way", "failed to yield ROW"
- "right of way violation", "disregarded the right of way"

Either party.

**Mark 0 in every other case**, however clearly the situation looks like a yield failure to you. This is the part coders get wrong, so read the list.

Each of these is 0, because another label already carries it or because it is simply the collision:

| What the narrative describes | Where it belongs |
|---|---|
| A vehicle turning across the rider's path | 5.9, not here |
| Running or disregarding a signal or stop sign | 5.5, not here |
| Entering from a driveway, alley, or parking lot | 5.3, not here |
| Moving from a sidewalk into the roadway | 5.2, not here |
| Failing to control speed, or losing control | 5.6, not here |
| "did not see", "did not observe" | 5.8, not here |
| "struck", "collided with", "crashed into", "crossed in front of", "came out of nowhere" | nowhere; that is the crash |
| A citation, fault finding, or statute reference with no yield language | nowhere (see section 4) |
| Responsibility stated to be undetermined | nowhere |

**Why the rule is this strict.** Every narrative you read ends in a collision, and in every one of them one party was in front of the other at the moment of impact. If you infer a yield failure from that, you will mark this label on almost every narrative and it will mean nothing. Mark it when the words are there.

**Examples.**
> "Unit 2 did not give Unit 1 the right of way." → **1**

> "Unit 1 said its light was green but did not yield to Unit 2 in the crosswalk." → **1**

> "Unit 2 did not yield at the yield sign." → **1**

> "Unit 1 turned left at the intersection and hit Unit 2, which was passing in front of it." → **0** (a vehicle turning across a path; that is 5.9)

> "Unit 1 did not stop at the stop sign and hit the scooter." → **0** (a signal violation; that is 5.5)

> "Unit 1 was going too fast and hit Unit 2 as Unit 2 came out of a private driveway." → **0** here (5.6 and 5.3 carry it)

**You will then be asked who.** When you mark this 1, a second question appears: **rider**, **driver**, **both**, or **unclear**. Answer from the narrative. "Both" is for narratives that describe each party failing to yield. "Unclear" is for narratives that use the phrase without saying who, or that give conflicting accounts.

> "Unit 1 possibly did not yield at the stop sign, and Unit 2 possibly did not yield to a vehicle." → 1, **both**

---

### 5.5 Signal or stop-sign violation

**What it is.** The narrative describes a party disregarding a traffic control device.

**Counts:**
- running or disregarding a red light, stop sign, stop-and-go signal, or "no walking" pedestrian signal
- "bypassed a stop sign", "failed to stop at the designated stop sign", "rolled through the stop sign"
- either party

**Does not count:**
- having a green light, or stopping properly (those are compliance)
- a location merely being signalized or stop-controlled
- failing to yield where no device is described (that is 5.4)

**Examples.**
> "Unit 1 entered the intersection on a red signal and hit Unit 2." → **1**

> "Unit 2 rode into the road without looking, against the don't walk signal." → **1**

> "Unit 1 halted at the stop sign, checked both directions and thought it was clear before turning left." → **0**

**You will then be asked who**, with the same four options as 5.4.

**Hard case.** "Crept past the stop line to get a better view" is a **1** (the stop was not made as required). Flag it.

---

### 5.6 Swerve or loss of control

**What it is.** The scooter swerved, slid, braked hard, or could not stop or steer as intended.

**Counts:**
- "lost control", "unable to stop", "could not stop in time", "failed to control speed"
- swerving, sliding, evasive action that failed
- falling off the scooter **before** any impact

**Does not count:**
- being knocked off or falling **as a result of** being struck (that is the crash, not a loss of control before it)
- the motor vehicle braking or losing control

**Examples.**
> "Unit 2 tried to swerve away, lost control and slid into the side of Unit 1." → **1**

> "Unit 2 said the scooter could not be stopped in time to avoid Unit 1." → **1**

> "The collision threw the young rider off the scooter onto the pavement." → **0**

---

### 5.7 Struck by an opening vehicle door

**What it is.** A door of a parked or stopped vehicle was opened into the scooter's path, and the scooter struck it or swerved because of it.

**Counts:**
- a door being opened into the path of the rider
- "doored", "door was opened", "opened his door into"

**Does not count:**
- the scooter striking a door of a **moving** vehicle
- a door being named only as the point of impact

**Examples.**
> "Unit 2 hit the front passenger door of Unit 1 while Unit 1 was moving." → **0** (a moving vehicle; the door is the impact point, nothing was opened)

> "The parked driver swung the door open as the scooter passed, and the rider hit it." → **1**

This is rare. Most narratives mentioning a door are 0.

---

### 5.8 Distraction or impairment cue

**What it is.** The narrative states something about attention, phone use, alcohol, or drugs, for either party.

**Counts:**
- phone use, texting, navigating, reaching, looking away
- "inattention", "distracted", "not paying attention"
- alcohol or drug use, intoxication, smell of alcohol, open container, a field sobriety test
- "did not see" or "did not observe" the other unit **when the narrative presents it as a failure to look**

**Does not count:**
- "came out of nowhere" said by a driver (a claim about the other party, not about attention)
- a blocked view or obstruction
- speed alone

**Examples.**
> "The driver of Unit 1 was looking at a phone for directions." → **1**

> "Unit 2 never saw Unit 1 and ran into its side." → **1**

> "The driver of Unit 1 said the scooter appeared out of nowhere." → **0**

**Hard case.** "Did not see" appears constantly. Code 1 when the narrative attributes it to the party's own looking or attention ("did not look", "did not observe", "failed to see"). Code 0 when it reports only that a party did not perceive the other in passing, with no attention framing. Flag either way when close.

---

### 5.9 Motor vehicle turning across the rider's path

**What it is.** The motor vehicle was turning, and that turn brought it across the scooter's line of travel.

**Counts:**
- the vehicle turning left or right into, across, or in front of the scooter's path
- turning into a driveway or side street across the scooter's direction
- a U-turn across the path

**Does not count:**
- the vehicle traveling straight
- the **scooter** turning
- a turn described somewhere in the narrative that plainly did not involve the scooter

**Examples.**
> "Unit 1 turned right into a driveway without yielding to Unit 2." → **1**

> "Unit 1 turned left and hit Unit 2, which was riding across the road." → **1**

> "Unit 1 was going straight west, and Unit 2 did not yield to Unit 1." → **0**

---

### 5.10 Rider lane positioning

**What it is.** The narrative says where in the roadway the scooter was.

**Counts:**
- in a bike lane, travel lane, shoulder, median, or between parked cars and the travel lane
- "in the right lane", "in lane 2", "on the improved shoulder"

**Does not count:**
- on the sidewalk (that is 5.2 territory, and sidewalks are not roadway lanes)
- crossing the road, where no lane position is given
- the motor vehicle's lane position

**Examples.**
> "Unit 2 rode an e-scooter north in the bike lane behind Unit 1." → **1**

> "Unit 2 rode an e-scooter in the gap between the parked cars and the travel lane." → **1**

**Note.** This label is about information being present, not about the position being wrong.

---

## 6. The five injury labels

These are different in kind. Here you look **only for explicit statements about injury**, and you read the narrative exactly as written.

A narrative can be silent about injury entirely. That is common and correct: all five get 0.

---

### 6.1 Transported for care

**1 when** the narrative states the rider was taken to a hospital, emergency room, or medical facility, by ambulance, EMS, helicopter, private vehicle, or on foot by a family member.

**0 when** transport was offered and refused, when the rider was only checked or evaluated on scene, or when transport is not mentioned.

> "An ambulance took Unit 2 to a hospital." → **1**
> "Unit 2 was hurt but declined to be taken to a hospital." → **0**
> "The rider's parents drove him to the hospital." → **1**

---

### 6.2 Unconscious or unresponsive

**1 when** the narrative states the rider lost consciousness, was unresponsive, was knocked out, or was found unconscious.

**0 when** the rider is described as awake, alert, dazed, or confused, or when consciousness is not mentioned.

> "Unit 2 was conscious and talking but in pain." → **0**

---

### 6.3 Fracture or severe laceration

**1 when** the narrative names a broken bone, a fracture, or a laceration described as severe, deep, or requiring sutures. Also 1 for amputation or a named internal injury.

**0 when** the narrative reports scrapes, abrasions, road rash, bruising, swelling, a minor or light laceration, or a bloody lip. Also 0 when it states the absence of fracture.

> "Unit 2 has no broken bones so far, but both legs are swollen and scraped." → **0**
> "Unit 2 has a small cut on one hand." → **0**
> "The rider had a serious injury to the right leg." → **0 and flag** (severe, but no fracture or laceration named)

---

### 6.4 Death

**1 when** the narrative states the rider died, was pronounced dead, was fatally injured, or is referred to as deceased.

**0** otherwise, including for injuries described as life-threatening or critical.

---

### 6.5 Explicit statement of no injury

**1 when** the narrative explicitly states that the rider was not injured, sustained no injuries, refused treatment or medical attention, said they were fine, or was cleared on scene.

**0 when** the narrative is silent about injury. **Silence is not a statement of no injury.**

> "EMS checked both riders of Unit 2, and both declined treatment." → **1**
> "Unit 1 stopped to ask whether Unit 2 was all right, and Unit 2 said yes." → **1**
> A narrative that describes the collision and stops → **0** on all five

**A narrative can be 1 on both 6.1 and 6.5** if it says, for example, that the rider stated no injury but was transported as a precaution. Code both as written.

---

## 7. Practical matters

**Order.** Narratives appear in a different random order for each coder. Code them in the order shown.

**Length.** Some narratives are two sentences, some are two pages with officer commentary, statute citations, and equipment serial numbers. Read the whole thing; code only the crash.

**Contradictions.** Some narratives contain an original account and a later correction by a detective. Code the corrected account and flag.

**Saving.** Your work saves in the browser as you go. At the end of every sitting press **Export CSV** and keep the file. If your browser data is cleared, the export is the only record.

**Questions.** Send them to the study coordinator, who will answer questions about the guide but will not tell you how to code a specific narrative.

**Deadline.** Batch 1 (150 narratives) by the evening of September 16. Batch 2 is a shorter second set, opened on September 17 with a new passcode.

---

*End of coding guide, version 1.2.*
