using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class FloaterContainer : MonoBehaviour
{
    [HideInInspector]
    public float submerged_buoyant_strength;
    private Collider m_collider;
    public List<GameObject> floaters = new List<GameObject>();
    public int num_floaters = 4;
    public Collider ocean_collider;
    private const float displacement_volume = 375;      // cubic inches
    private float net_buoyancy;      // cubic inches
    void Start()
    {
        /* Fb = pgV, volume of rover from https://discuss.bluerobotics.com/t/volume-of-bluerov/245/5  NEED TO TEST VOLUME IN REAL */
        //submerged_buoyant_strength = Utils.RHO_SALT_WATER * -Physics.gravity.y * (displacement_volume * Utils.metric_conversion_constant);

        /* Default net buoyancy from https://bluerobotics.com/wp-content/uploads/2020/02/br_bluerov2_datasheet_rev6.pdf */
        net_buoyancy = GetComponent<Rigidbody>().mass > 11 ? 0.2f : 1.4f;
        submerged_buoyant_strength = (net_buoyancy * -Physics.gravity.y) + (GetComponent<Rigidbody>().mass * -Physics.gravity.y);

        m_collider = GetComponent<Collider>();
        InitialiseFloaters();
    }

    public void InitialiseFloaters()
    {
        for (int i = 0; i < num_floaters; ++i)
        {
            GameObject new_floater = new GameObject("Floater_" + i.ToString());
            new_floater.tag = "Floater";
            new_floater.transform.parent = transform;

            switch (i)
            {
                case 0:
                    new_floater.transform.position = transform.position + new Vector3(m_collider.bounds.extents.x, 0, m_collider.bounds.extents.z);
                    break;
                case 1:
                    new_floater.transform.position = transform.position + new Vector3(-m_collider.bounds.extents.x, 0, -m_collider.bounds.extents.z);
                    break;
                case 2:
                    new_floater.transform.position = transform.position + new Vector3(m_collider.bounds.extents.x, 0, -m_collider.bounds.extents.z);
                    break;
                case 3:
                    new_floater.transform.position = transform.position + new Vector3(-m_collider.bounds.extents.x, 0, m_collider.bounds.extents.z);
                    break;
            }

            new_floater.AddComponent<SphereCollider>();
            new_floater.AddComponent<Floater>();
            new_floater.GetComponent<Floater>().buoyant_strength = submerged_buoyant_strength / num_floaters;
            new_floater.GetComponent<Floater>().water_collider = ocean_collider;
            floaters.Add(new_floater);
        }
    }

    private void FixedUpdate()
    {
    }
}
