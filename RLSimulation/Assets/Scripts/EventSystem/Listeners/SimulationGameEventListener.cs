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

    [SerializeField]
    public JsonServerConfigUnityEvent server_config_received_unity_event;

    [SerializeField]
    public ServerConnectingUnityEvent server_connecting_unity_event;

    [SerializeField]
    public ExceptionUnityEvent server_connection_attempted_unity_event;

    [SerializeField]
    public JsonControlUnityEvent json_control_unity_event;

    [SerializeField]
    public UnityEvent observation_sent_unity_event;

    public override void OnEnable()
    {
        game_events.Add(EventMaster._instance.reset_episode_event);
        game_events.Add(EventMaster._instance.end_simulation_event);
        game_events.Add(EventMaster._instance.rov_initialised_event);
        game_events.Add(EventMaster._instance.server_config_received_event);
        game_events.Add(EventMaster._instance.server_connecting_event);
        game_events.Add(EventMaster._instance.server_connection_attempt_event);
        game_events.Add(EventMaster._instance.json_control_event);
        game_events.Add(EventMaster._instance.observation_sent_event);
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

    public void OnServerConfigReceivedEventRasied(JsonMessage<ServerConfig> param)
    {
        server_config_received_unity_event.Invoke(param);
    }

    public void OnServerConnectingEventRaised(string ip, int port)
    {
        server_connecting_unity_event.Invoke(ip, port);
    }

    public void OnServerConnectedAttemptEventRaised(Exception e)
    {
        server_connection_attempted_unity_event.Invoke(e);
    }

    public void OnJsonControlEventRaised(JsonMessage<JsonControls> msg)
    {
        json_control_unity_event.Invoke(msg);
    }

    public void OnObservationSentEventRaised()
    {
        observation_sent_unity_event.Invoke();
    }
}