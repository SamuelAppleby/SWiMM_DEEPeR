using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static Server;

[CreateAssetMenu]
public class JsonControlsEvent : GameEvent
{
    public void Raise(JsonMessage param)
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnJsonControlEventRaised(param);
        }
    }
}