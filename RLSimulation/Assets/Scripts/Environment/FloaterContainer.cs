using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class FloaterContainer : MonoBehaviour
{
    [HideInInspector]
    public float total_buoyant_strength;
    private Collider m_collider;
    private List<GameObject> floaters = new List<GameObject>();
    [HideInInspector]
    public bool is_initialized = false;
    private const int NUM_FLOATERS = 4;
    public Collider ocean_collider;

    void Start()
    {
        /* Fb = pgV, approximation of box */
        total_buoyant_strength = Utils.RHO_WATER * -Physics.gravity.y * Utils.VolumeOfBoxCollider(GetComponent<BoxCollider>());
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
            for (int i = 0; i < NUM_FLOATERS; ++i)
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
                new_floater.AddComponent<Floater>().buoyant_strength = total_buoyant_strength / NUM_FLOATERS;
                new_floater.GetComponent<Floater>().water_collider = ocean_collider;
                floaters.Add(new_floater);
            }
        }
    }
}
