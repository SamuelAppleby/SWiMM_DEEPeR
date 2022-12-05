using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;
using UnityEngine.UIElements;

public class MainMenuGameEventListener : GameEventListener
{
    [SerializeField]
    public ServerConnectingUnityEvent server_connecting_unity_event;

    [SerializeField]
    public ExceptionUnityEvent server_connection_attempted_unity_event;

    [SerializeField]
    public UnityEvent awaiting_training_unity_event;

    public override void OnEnable()
    {
        game_events.Add(EventMaster._instance.server_connecting_event);
        game_events.Add(EventMaster._instance.server_connection_attempt_event);
        game_events.Add(EventMaster._instance.server_awaiting_training_event);
        base.OnEnable();
    }

    protected override void OnDisable()
    {
        base.OnDisable();
    }

    public void OnServerAwaitingTrainingEventRaised()
    {
        awaiting_training_unity_event.Invoke();
    }

    public void OnServerConnectingEventRaised(string ip, int port)
    {
        server_connecting_unity_event.Invoke(ip, port);
    }

    public void OnServerConnectedAttemptEventRaised(Exception e)
    {
        server_connection_attempted_unity_event.Invoke(e);
    }
}