using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Server;

[CreateAssetMenu]
public class ServerConfigReceivedEvent : GameEvent
{
    public void Raise(JsonMessage<ServerConfig> param)
    {
        foreach (SimulationGameEventListener l in listeners)
        {
            l.OnServerConfigReceivedEventRasied(param);
        }
    }
}
