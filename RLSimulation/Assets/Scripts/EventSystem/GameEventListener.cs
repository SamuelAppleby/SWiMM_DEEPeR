using UnityEngine.Events;
using UnityEngine;
using System.Collections.Generic;
using System;
using static Server;
using UnityEngine.UIElements;

public class GameEventListener : MonoBehaviour
{
    public List<GameEvent> game_events = new List<GameEvent>();

    public virtual void OnEnable()
    {
        foreach(GameEvent e in game_events)
        {
            e.RegisterListener(this);
        }
    }

    protected virtual void OnDisable()
    {
        foreach (GameEvent e in game_events)
        {
            e.UnregisterListener(this);
        }
    }
}