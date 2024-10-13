#define EPSILON (1e-9f)
#define BLOCK_DIM (128)

/*
* @brief calculate velocity of particles
* @param pi particle i
* @param pj particle j
* @param ai force of i
*/
__kernel void dCalcVelocity(float t, int size, __global float4* pos, __global float4* vel, __local float4* shared_pos)
{
    int globalIdx = get_global_id(0);

    if (globalIdx < size) {
        float3 f = {0.0f, 0.0f, 0.0f};

        for (int tileIdx = 0; tileIdx < get_num_groups(0); ++tileIdx) {
            float4 temp = pos[tileIdx * get_local_size(0) + get_local_id(0)];
            shared_pos[get_local_id(0)].x = temp.x;
            shared_pos[get_local_id(0)].y = temp.y;
            shared_pos[get_local_id(0)].z = temp.z;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < BLOCK_DIM; ++i) {
                float dist_x = shared_pos[i].x - pos[globalIdx].x;
                float dist_y = shared_pos[i].y - pos[globalIdx].y;
                float dist_z = shared_pos[i].z - pos[globalIdx].z;

                float distSqr = pow(dist_x, 2) + pow(dist_y, 2) + pow(dist_z, 2) + EPSILON;
                float invDist = rsqrt(distSqr);
                float invDistCube = pow(invDist, 3);

                f.x += dist_x * invDistCube;
                f.y += dist_y * invDistCube;
                f.z += dist_z * invDistCube;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        vel[globalIdx].x = t * f.x;
        vel[globalIdx].y = t * f.y;
        vel[globalIdx].z = t * f.z;
    }
}