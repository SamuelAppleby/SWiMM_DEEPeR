using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Server;

[CreateAssetMenu]
public class ServerEvent : GameEvent
{
    public void Raise(JsonMessage param)
    {
        foreach (GameEventListener l in listeners)
        {
            switch (param.msgType)
            {
                case "process_server_config":
                    l.OnServerConfigReceivedEventRasied(param);
                    break;
                case "reset_episode":
                    l.OnEpisodeResetEventRaised();
                    break;
                case "receive_json_controls":
                    l.OnJsonControlEventRaised(param);
                    break;
                case "end_simulation":
                    l.OnSimulationEndEventRaised();
                    break;
                case "set_position":
                    l.OnSetPositionEventRaised(param);
                    break;
                default:
                    break;
            }
        }
    }
}