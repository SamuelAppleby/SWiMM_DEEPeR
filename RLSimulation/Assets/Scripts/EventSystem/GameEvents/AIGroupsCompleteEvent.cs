using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu]
public class AIGroupsCompleteEvent : GameEvent
{
    public void Raise()
    {
        foreach (PlayerGameEventListener l in listeners)
        {
            l.OnAIGroupsCompleteEventRaised();
        }
    }
}
