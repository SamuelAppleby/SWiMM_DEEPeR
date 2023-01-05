using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DolphinAnimation : MonoBehaviour
{
    Animator dolphinAnimator;

    public float forwardSpeed = 1f;
    public float turnSpeed = .0f;
    public float upDownSpeed = 0f;
    public float rollSpeed = 0f;

    void Start()
    {
        dolphinAnimator = GetComponent<Animator>();
    }

    void FixedUpdate()
    {
        Move();
    }

    public void Move()
    {
        dolphinAnimator.SetFloat("Forward", forwardSpeed);
        dolphinAnimator.SetFloat("Turn", turnSpeed);
        dolphinAnimator.SetFloat("UpDown", upDownSpeed);
        dolphinAnimator.SetFloat("Roll", rollSpeed);
    }
}
