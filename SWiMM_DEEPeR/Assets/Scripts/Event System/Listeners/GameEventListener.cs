using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;

public class GameEventListener : MonoBehaviour
{
    [HideInInspector]
    public List<GameEvent> game_events = new List<GameEvent>();

    [SerializeField]
    public ServerConnectingUnityEvent server_connecting_unity_event;

    [SerializeField]
    public ExceptionUnityEvent server_connection_attempted_unity_event;

    [SerializeField]
    public JsonDataUnityEvent server_config_received_unity_event;

    [SerializeField]
    public UnityEvent reset_episode_unity_event;

    [SerializeField]
    public ROVInitialisedUnityEvent rov_initialised_unity_event;

    [SerializeField]
    public JsonDataUnityEvent json_control_unity_event;

    [SerializeField]
    public UnityEvent observation_sent_unity_event;

    [SerializeField]
    public UnityEvent end_simulation_unity_event;

    [SerializeField]
    public UnityEvent ai_groups_complete_unity_event;

    [SerializeField]
    public SceneChangedUnityEvent scene_changed_unity_event;

    [SerializeField]
    public JsonDataUnityEvent set_position_unity_event;

    public virtual void OnEnable()
    {
        SimulationManager._instance.event_master.rov_initialised_event.RegisterListener(this);
        SimulationManager._instance.event_master.server_connecting_event.RegisterListener(this);
        SimulationManager._instance.event_master.server_connection_attempt_event.RegisterListener(this);
        SimulationManager._instance.event_master.observation_sent_event.RegisterListener(this);
        SimulationManager._instance.event_master.ai_groups_complete_event.RegisterListener(this);
        SimulationManager._instance.event_master.scene_changed_event.RegisterListener(this);
        SimulationManager._instance.event_master.server_event.RegisterListener(this);
        SimulationManager._instance.event_master.sent_event.RegisterListener(this);
    }

    protected virtual void OnDisable()
    {
        SimulationManager._instance.event_master.rov_initialised_event.UnregisterListener(this);
        SimulationManager._instance.event_master.server_connecting_event.UnregisterListener(this);
        SimulationManager._instance.event_master.server_connection_attempt_event.UnregisterListener(this);
        SimulationManager._instance.event_master.observation_sent_event.UnregisterListener(this);
        SimulationManager._instance.event_master.ai_groups_complete_event.UnregisterListener(this);
        SimulationManager._instance.event_master.scene_changed_event.UnregisterListener(this);
        SimulationManager._instance.event_master.server_event.UnregisterListener(this);
        SimulationManager._instance.event_master.sent_event.UnregisterListener(this);
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

    public void OnServerConfigReceivedEventRasied(JsonMessage param)
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

    public void OnJsonControlEventRaised(JsonMessage msg)
    {
        json_control_unity_event.Invoke(msg);
    }

    public void OnObservationSentEventRaised()
    {
        observation_sent_unity_event.Invoke();
    }

    public void OnAIGroupsCompleteEventRaised()
    {
        ai_groups_complete_unity_event.Invoke();
    }

    public void OnSceneChangedEventRaised(Enums.E_SceneIndices to)
    {
        scene_changed_unity_event.Invoke(to);
    }

    public void OnSetPositionEventRaised(JsonMessage msg)
    {
        set_position_unity_event.Invoke(msg);
    }
}