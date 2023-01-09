using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Server;

[CreateAssetMenu]
public class SentEvent : GameEvent
{
    public void Raise(DataToSend param)
    {
        foreach (GameEventListener l in listeners)
        {
            switch (param.msg_type)
            {
                case "connection_request":
                    break;
                case "on_server_config_received":
                    break;
                case "training_ready":
                    break;
                case "on_telemetry":
                    l.OnObservationSentEventRaised();
                    break;
                default:
                    break;
            }
        }
    }
}