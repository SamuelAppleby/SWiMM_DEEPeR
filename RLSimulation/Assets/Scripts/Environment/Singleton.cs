using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Singleton<T> : MonoBehaviour where T : Component
{
    public static T _instance { get; private set; }

    protected virtual void Awake()
    {
        if (_instance == null)
        {
            _instance = this as T; // In first scene, make us the singleton.
            DontDestroyOnLoad(gameObject);
        }
        else if (_instance != this)
        {
            Destroy(gameObject); // On reload, singleton already set, so destroy duplicate.
        }

        OnSceneChanged();
    }

    protected virtual void OnSceneChanged()
    {

    }
}
