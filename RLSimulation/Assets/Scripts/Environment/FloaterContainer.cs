using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class FloaterContainer : MonoBehaviour
{
    [Header("Total buoyant strength (N) of ALL colliders, distrubted evenly")]
    public float total_buoyant_strength = 300;
    private Collider m_collider;
    private List<GameObject> floaters = new List<GameObject>();

    void Start()
    {
        m_collider = GetComponent<Collider>();
        InitialiseFloaters();
    }

    public void InitialiseFloaters()
    {
        BoxCollider col = m_collider as BoxCollider;
        if (col)
        {
            for (int i = 0; i < 4; ++i)
            {
                GameObject new_floater = new GameObject("Floater_" + i.ToString());
                new_floater.tag = "Floater";
                new_floater.transform.parent = transform;

                if (i == 0)
                {
                    new_floater.transform.position = transform.position + new Vector3(col.size.x / 2, 0, 0);
                }
                else if (i == 1)
                {
                    new_floater.transform.position = transform.position + new Vector3(-col.size.x / 2, 0, 0);
                }
                else if (i == 2)
                {
                    new_floater.transform.position = transform.position + new Vector3(0, 0, col.size.z / 2);
                }
                else if (i == 3)
                {
                    new_floater.transform.position = transform.position + new Vector3(0, 0, -col.size.z / 2);
                }

                new_floater.AddComponent<SphereCollider>();
                new_floater.AddComponent<Floater>().buoyant_strength = total_buoyant_strength / 4;
                floaters.Add(new_floater);
            }
        }
    }
}
