using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;
using UnityEngine.UIElements;

public class PlayerGameEventListener : GameEventListener
{
    [SerializeField]
    public JsonControlUnityEvent json_control_unity_event;

    [SerializeField]
    public UnityEvent ai_groups_complete_unity_event;

    public override void OnEnable()
    {
        game_events.Add(EventMaster._instance.json_control_event);
        game_events.Add(EventMaster._instance.ai_groups_complete_event);
        base.OnEnable();
    }

    protected override void OnDisable()
    {
        base.OnDisable();
    }

    public void OnJsonControlEventRaised(JsonMessage<JsonControls> msg)
    {
        json_control_unity_event.Invoke(msg);
    }

    public void OnAIGroupsCompleteEventRaised()
    {
        ai_groups_complete_unity_event.Invoke();
    }
}