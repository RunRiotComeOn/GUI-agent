# Online Change Memory Template

Use this template to manually maintain rolling memory during long GUI
interaction trajectories.

## Current Memory

```text
[older_summary]
<empty or previous older summary>

[recent_buffer]
<recent items in time order>
```

## New Step Update

```text
[compare]
previous visible state: ...
current visible state: ...

[decision]
item_type: recent_change | recent_change_keyframe
reason: ...
```

## Required Anchors For Every Item

```text
element: <acted-on control or current key target>
action_type: CLICK | SELECT | TYPE | ""
action_value: <typed/selected value or empty>
focus_after: <active UI sub-flow after this step>
next_goal: <next immediate local target, not the whole task>
```

## If `recent_change`

```text
[recent_change]
change: ...
element: ...
action_type: ...
action_value: ...
focus_after: ...
next_goal: ...
```

## If `recent_change_keyframe`

```text
[recent_change_keyframe]
change: ...
element: ...
action_type: ...
action_value: ...
focus_after: ...
next_goal: ...
action: ...
image: keep
```

## If Buffer Too Long

Summarize the older prefix only:

```text
[older_summary_update]
<new higher-level summary of older prefix>
```

Then keep only the newest few recent items verbatim.

Important:

- do not summarize away branch-defining focus like `location continuation` vs
  `add-ons`
- do not summarize away exact input actions like `TYPE email`
- when one branch becomes active and another branch is no longer relevant, say
  that explicitly

## Final Context Shape

```text
[older_summary]
...

[recent_buffer]
1. [recent_change] ... | element: ... | action_type: ... | action_value: ... | focus_after: ... | next_goal: ...
2. [recent_change_keyframe] ... | element: ... | action_type: ... | action_value: ... | focus_after: ... | next_goal: ... | action: ...
3. [recent_change] ... | element: ... | action_type: ... | focus_after: ... | next_goal: ...

[current_image]
<current screenshot>
```
