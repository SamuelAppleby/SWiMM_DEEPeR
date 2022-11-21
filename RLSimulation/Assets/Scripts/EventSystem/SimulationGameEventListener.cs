using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;
using UnityEngine.UIElements;

public class SimulationGameEventListener : GameEventListener
{
    [SerializeField]
    public UnityEvent reset_episode_unity_event;

    [SerializeField]
    public UnityEvent end_simulation_unity_event;

    [SerializeField]
    public ROVInitialisedUnityEvent rov_initialised_unity_event;

    public override void OnEnable()
    {
        game_events.Add(EventMaster._instance.reset_episode_event);
        game_events.Add(EventMaster._instance.end_simulation_event);
        game_events.Add(EventMaster._instance.rov_initialised_event);
        base.OnEnable();
    }

    protected override void OnDisable()
    {
        base.OnDisable();
    }

    public void OnEpisodeResetEventRaised()
    {
        reset_episode_unity_event.Invoke();
    }

    public void OnSimulationEndEventRaised()
    {
        end_simulation_unity_event.Invoke();
    }

    public void OnROVInitalisedEventRaised(GameObject obj)
    {
        rov_initialised_unity_event.Invoke(obj);
    }
}