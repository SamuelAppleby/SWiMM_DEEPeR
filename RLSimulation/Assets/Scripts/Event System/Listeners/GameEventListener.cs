using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;
using UnityEngine.UIElements;

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
    public UnityEvent awaiting_training_unity_event;

    [SerializeField]
    public UnityEvent ai_groups_complete_unity_event;

    [SerializeField]
    public SceneChangedUnityEvent scene_changed_unity_event;

    public virtual void OnEnable()
    {
        EventMaster._instance.reset_episode_event.RegisterListener(this);
        EventMaster._instance.end_simulation_event.RegisterListener(this);
        EventMaster._instance.rov_initialised_event.RegisterListener(this);
        EventMaster._instance.server_config_received_event.RegisterListener(this);
        EventMaster._instance.server_connecting_event.RegisterListener(this);
        EventMaster._instance.server_connection_attempt_event.RegisterListener(this);
        EventMaster._instance.json_control_event.RegisterListener(this);
        EventMaster._instance.observation_sent_event.RegisterListener(this);
        EventMaster._instance.server_awaiting_training_event.RegisterListener(this);
        EventMaster._instance.ai_groups_complete_event.RegisterListener(this);
        EventMaster._instance.scene_changed_event.RegisterListener(this);
    }

    protected virtual void OnDisable()
    {
        EventMaster._instance.reset_episode_event.UnregisterListener(this);
        EventMaster._instance.end_simulation_event.UnregisterListener(this);
        EventMaster._instance.rov_initialised_event.UnregisterListener(this);
        EventMaster._instance.server_config_received_event.UnregisterListener(this);
        EventMaster._instance.server_connecting_event.UnregisterListener(this);
        EventMaster._instance.server_connection_attempt_event.UnregisterListener(this);
        EventMaster._instance.json_control_event.UnregisterListener(this);
        EventMaster._instance.observation_sent_event.UnregisterListener(this);
        EventMaster._instance.server_awaiting_training_event.UnregisterListener(this);
        EventMaster._instance.ai_groups_complete_event.UnregisterListener(this);
        EventMaster._instance.scene_changed_event.UnregisterListener(this);
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

    public void OnServerAwaitingTrainingEventRaised()
    {
        awaiting_training_unity_event.Invoke();
    }

    public void OnAIGroupsCompleteEventRaised()
    {
        ai_groups_complete_unity_event.Invoke();
    }

    public void OnSceneChangedEventRaised(Enums.E_SceneIndices to)
    {
        scene_changed_unity_event.Invoke(to);
    }
}