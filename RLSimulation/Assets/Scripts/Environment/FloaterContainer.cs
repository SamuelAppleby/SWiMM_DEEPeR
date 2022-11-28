using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class FloaterContainer : MonoBehaviour
{
    [Header("Total buoyant strength (N) of ALL colliders, distrubted evenly")]
    public float total_buoyant_strength;
    private Collider m_collider;
    private List<GameObject> floaters = new List<GameObject>();
    [HideInInspector]
    public bool is_initialized = false;

    void Start()
    {
        if (SimulationManager._instance.server != null && SimulationManager._instance.server.json_server_config.msgType.Length > 0)
        {
            total_buoyant_strength = SimulationManager._instance.server.json_server_config.payload.serverConfig.roverConfig.structureConfig.totalBuoyantForce;
        }

        m_collider = GetComponent<Collider>();
        InitialiseFloaters();
        is_initialized = true;
    }

    public void InitialiseFloaters()
    {
        Vector2 extents = Vector2.zero;

        extents.x = m_collider.bounds.extents.x;
        extents.y = m_collider.bounds.extents.y;

        if (extents != Vector2.zero)
        {
            for (int i = 0; i < 4; ++i)
            {
                GameObject new_floater = new GameObject("Floater_" + i.ToString());
                new_floater.tag = "Floater";
                new_floater.transform.parent = transform;

                if (i == 0)
                {
                    new_floater.transform.position = transform.position + new Vector3(extents.x / 2, 0, 0);
                }
                else if (i == 1)
                {
                    new_floater.transform.position = transform.position + new Vector3(-extents.x / 2, 0, 0);
                }
                else if (i == 2)
                {
                    new_floater.transform.position = transform.position + new Vector3(0, 0, extents.y / 2);
                }
                else if (i == 3)
                {
                    new_floater.transform.position = transform.position + new Vector3(0, 0, -extents.y / 2);
                }

                new_floater.AddComponent<SphereCollider>();
                new_floater.AddComponent<Floater>().buoyant_strength = total_buoyant_strength / 4;
                floaters.Add(new_floater);
            }
        }
    }
}
