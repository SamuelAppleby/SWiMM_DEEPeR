using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;
using UnityEngine.UIElements;

public class FishSpawnerGameEventListener : GameEventListener
{
    [SerializeField]
    public ROVInitialisedUnityEvent rov_initialised_unity_event;

    public override void OnEnable()
    {
        game_events.Add(EventMaster._instance.rov_initialised_event);
        base.OnEnable();
    }

    protected override void OnDisable()
    {
        base.OnDisable();
    }

    public void OnROVInitalisedEventRaised(GameObject obj)
    {
        rov_initialised_unity_event.Invoke(obj);
    }
}