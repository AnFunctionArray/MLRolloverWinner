//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

float4 RotateAroundYInDegrees(float4 vertex, float degrees)
{
    float alpha = degrees * 3.14159265f / 180.0;
    float sina, cosa;
    sincos(alpha, sina, cosa);
    float2x2 m = float2x2(cosa, -sina, sina, cosa);
    return float4(mul(m, vertex.xz), vertex.yw).xzyw;
}

PSInput VSMain(float4 position : POSITION, float4 color : COLOR)
{
    PSInput result;

    position = RotateAroundYInDegrees(position, color.w);

    result.position = position;
    result.color = color;

    return result;
}
/*
#define OutputSHeapMaX 14

static PSInput OutputSHeap[OutputSHeapMaX];

static uint OutputSHeapSize = 0;

[instance(OutputSHeapMaX)]
[maxvertexcount(3)]
void GSMain(triangle PSInput input[3], inout TriangleStream<PSInput> OutputStream, uint InstanceID : SV_GSInstanceID)
{
    if ((InstanceID * 3) == OutputSHeapSize && !donestore) {
        for (uint i = 0; i < 3; ++i) {
            OutputSHeap[i + OutputSHeapSize] = input[i];
            OutputStream.Append(input[i]);
        }
        OutputSHeapSize += 3;

        if (OutputSHeapSize >= OutputSHeapMaX) {
            OutputSHeapSize = 0;
        }

        donestore = true;
    }
    else {
        for (uint i = 0; i < 3; ++i) {
            OutputStream.Append(OutputSHeap[InstanceID + i]);
        }
    }
}
*/

float4 PSMain(PSInput input) : SV_TARGET
{

    return input.color;
}