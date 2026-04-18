# Online Change Memory Prompt

Use this prompt to maintain a rolling, hierarchical memory for long-horizon GUI
interaction trajectories.

This prompt is designed for the setting where the full history cannot remain in
context forever. Memory must be updated online after each new step.

## Core Principle

Do **not** rebuild a global summary from the whole trajectory every time.

Instead, for each new step:

1. First extract the newest local change as either:
   - `recent_change`
   - `recent_change_keyframe`
2. Append that new change item to recent memory
3. Only then, if memory is getting too long, summarize the **older prefix before
   the newest change items**

So the order is:

`extract newest change -> append -> summarize older prefix if needed`

The memory must preserve **action grounding**, not only workflow narration.

That means each update should keep:

- what element was acted on
- what action primitive happened: `CLICK`, `SELECT`, or `TYPE`
- what value was typed or selected if any
- what UI sub-flow is active after the step
- what the next immediate local goal is

Without these anchors, memory often becomes too abstract and the task agent
confuses nearby branches such as:

- `truck options` vs `add-ons`
- `location continuation` vs `choose add-ons`
- `guest modal` vs `results page`
- `payment options` vs `personal info form`

## Memory Structure

The memory is divided into two levels:

### 1. `older_summary`

A compressed summary of older interaction history.

Properties:

- coarse-grained
- no detailed step-by-step narration
- may summarize many earlier local changes
- should capture major task progress and major state transitions only

### 2. `recent_buffer`

A short ordered list of the most recent fine-grained memory items.

Each item is either:

- `recent_change`
- `recent_change_keyframe`

## Item Types

### `recent_change`

Use this when the new step causes only a local or incremental visible update.

Examples:

- a field becomes filled
- a date gets selected
- a filter value changes
- a small panel state changes
- the same page remains active and only local controls update

Format:

```text
[recent_change]
prev -> current: <very short visible state change>
```

Keep it extremely short:

- one short sentence
- or one compact clause
- but do not omit the interaction anchor fields described below

Examples:

- `prev -> current: check-in date is now selected.`
- `prev -> current: the same results page remains, but the sort menu is open.`
- `prev -> current: the planner panel advances from destination typing to a selected destination state.`

### `recent_change_keyframe`

Use this when the new step creates a major state transition and should remain in
memory more explicitly.

Typical triggers:

- page navigation
- modal/dialog opening
- entering checkout/payment
- switching from search/list view to detail view
- a large layout or interaction-focus jump

Important:

- a keyframe is not only a page jump
- it should also be used when the active branch changes in a way that affects
  the next legal action
- for example, `truck options -> location continuation` is a keyframe-worthy
  focus shift even if the page still looks similar

## Required Interaction Anchors

Every new memory item should preserve these fields in addition to a very short
`change` text:

- `element`: the acted-on control or the most relevant UI target
- `action_type`: one of `CLICK`, `SELECT`, `TYPE`, or empty
- `action_value`: the selected/typed value if any
- `focus_after`: the active UI sub-flow after the step
- `next_goal`: the next immediate local target implied by the current UI

### Why These Fields Matter

They prevent the memory from collapsing into vague statements like:

- `the user moved further in checkout`
- `the process continued`
- `the next screen opened`

Those summaries are usually too weak for candidate grounding.

Instead, prefer memory that still distinguishes things like:

- `focus_after: truck options`
- `next_goal: click the correct Select Truck button`

or

- `focus_after: location continuation`
- `next_goal: continue within location flow, not add-ons`

or

- `focus_after: personal info form`
- `next_goal: type email into the required email input`

Format:

```text
[recent_change_keyframe]
change: <very short major state transition>
action: <action string if available>
image: <keep this step image in memory>
```

Examples:

- `change: moved from pass comparison to purchase modal.`
- `action: CLICK`
- `image: keep`

or

- `change: results view transitions into checkout summary.`
- `action: CLICK`
- `image: keep`

## Update Procedure

For each new step:

### Step 1. Compare only the newest transition

Compare:

- previous visible state
- current visible state

Do **not** revisit the full trajectory first.

### Step 2. Emit one new memory item

Decide whether the newest transition is:

- `recent_change`
- or `recent_change_keyframe`

Then append it to `recent_buffer`.

### Step 3. Check memory budget

If `recent_buffer` is still short enough:

- stop here

If it is too long:

- summarize the **older prefix** of `recent_buffer`
- keep only the newest few recent items in detailed form

Important:

- the newest recent items should remain detailed
- do not immediately collapse the newest change you just extracted

## How To Summarize Older Prefix

When summarizing older prefix items into `older_summary`:

- summarize only the older part, not the newest tail
- keep only major state progress
- merge repeated small local updates
- preserve important keyframe transitions at a coarse level
- do not abstract away branch-defining anchors
- keep exact sub-flow shifts when they disambiguate future actions
- keep important `TYPE` / `SELECT` values if later steps depend on them

The resulting `older_summary` should answer:

- what stable workspace the trajectory moved into
- what major state transitions happened afterward
- what broad interaction phase the user had already reached

### Good Older Summary Style

- `The flow moved from initial search setup into stable results browsing, then into the guest modal, and later back to results with filters applied.`
- `The user progressed from pass browsing into payment options, then into the personal info form where name fields were being filled.`
- `The truck flow moved from search setup into truck options, then shifted into location continuation rather than add-ons.`

### Bad Older Summary Style

- long chronological narration
- restating every tiny click
- repeating the full task text
- keeping low-level details that belong in recent memory instead
- vague phrases that erase the active branch, such as `moved forward`, `continued checkout`, or `next step opened`

## Compression Policy

Use these defaults:

- keep newest fine-grained items in `recent_buffer`
- compress older repeated local changes aggressively
- allow major keyframe transitions to survive longer than ordinary small changes

A reasonable policy is:

- ordinary local changes get summarized sooner
- keyframes remain in recent memory longer
- very old keyframes eventually collapse into `older_summary`

## Final Context Template

When constructing the model context, prefer this order:

```text
[older_summary]
<older compressed memory>

[recent_buffer]
<recent_change / recent_change_keyframe items in order>

[current_image]
<current screenshot>
```

If keyframe images are retained in recent memory, they should be placed near
their corresponding `recent_change_keyframe` entries.

Each recent item should look more like this:

```text
1. [recent_change] truck options appear with pricing | element: Find Your Truck button | action_type: CLICK | focus_after: truck options | next_goal: choose the correct truck card
2. [recent_change_keyframe] location step becomes active | element: Continue to Location button | action_type: CLICK | focus_after: location continuation | next_goal: continue within location flow, not add-ons | action: [button] Continue to Location -> CLICK
3. [recent_change] email field becomes filled | element: email input | action_type: TYPE | action_value: jame_jones@hotmail.com | focus_after: personal info form | next_goal: fill the next required input
```

## Minimal Decision Heuristic

If uncertain, use this rule:

- choose `recent_change` for small same-page updates
- choose `recent_change_keyframe` when the interaction focus or page state clearly shifts
- summarize only the older prefix, never the newest event first

## Short Operator Version

```text
For each new step:
1. Compare previous state vs current state
2. Write one new item: recent_change or recent_change_keyframe
3. Append it to recent_buffer
4. If memory is too long, summarize the older prefix into older_summary
5. Keep the newest few items detailed
```
