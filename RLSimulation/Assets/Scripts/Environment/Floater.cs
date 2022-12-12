using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Floater : MonoBehaviour
{
    public float buoyant_strength = 10;
    private Rigidbody rb;
    private SphereCollider m_collider;
    private Collider other_collier;

    void Start()
    {
        rb = GetComponentInParent<Rigidbody>();
        m_collider = GetComponent<SphereCollider>();
        m_collider.radius = .15f;
        m_collider.isTrigger = true;
    }

    private void OnCollisionStay(Collision other)
    {
        if (other.gameObject.tag.Equals("Water"))
        {
            other_collier = other.collider;
        }
    }

    private void OnCollisionExit(Collision other)
    {
        if (other.gameObject.tag.Equals("Water"))
        {
            other_collier = null;
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.gameObject.tag.Equals("Water"))
        {
            other_collier = other;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.tag.Equals("Water"))
        {
            other_collier = null;
        }
    }

    private void FixedUpdate()
    {
        if (other_collier != null)
        {
            BoxCollider other_as_box = other_collier as BoxCollider;
            float top_of_water = other_collier.transform.position.y + (other_as_box.size.y / 2);
            float bottom_of_collider = GetComponent<Collider>().transform.position.y - m_collider.radius;
            float diff = Math.Clamp(top_of_water - bottom_of_collider, 0, (2 * m_collider.radius));
            float floater_ratio_underwater = diff / (2 * m_collider.radius);
            rb.AddForceAtPosition(Vector3.up * floater_ratio_underwater * buoyant_strength, transform.position, ForceMode.Force);
        }
    }
}
