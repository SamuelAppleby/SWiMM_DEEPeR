using UnityEngine;
using static Server;

[CreateAssetMenu]
public class ServerConfigReceivedEvent : GameEvent
{
    public void Raise(JsonMessage param)
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnServerConfigReceivedEventRasied(param);
        }
    }
}
