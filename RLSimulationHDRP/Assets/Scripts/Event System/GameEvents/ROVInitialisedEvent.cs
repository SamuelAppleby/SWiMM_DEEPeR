using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class ROVInitialisedEvent : GameEvent
{
    public void Raise(GameObject obj)
    {
        foreach (GameEventListener l in listeners)
        {
            l.OnROVInitalisedEventRaised(obj);
        }
    }
}
