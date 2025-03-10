# Wan Generation API Documentation

## Overview
The Wan Generation API provides endpoints for text-to-video, text-to-image, and image-to-video generation using the Wan model. The service runs on port 6006 and requires GPU availability for generation tasks.

## Base URL

http://localhost:6006

## Endpoints

### Check Service Status

GET /status

Returns the current status of the service including GPU availability and loaded models.

#### Response

```json
{
    "status": "running",
    "gpu_available": true,
    "device": 0,
    "loaded_pipelines": ["t2v-14B", "i2v-14B"]
}
```

### Text-to-Video/Image Generation

POST /generate/text

Generates a video or image from a text prompt.

#### Request Body

```json
{
    "task": string,          // Required: "t2v-14B", "t2v-1.3B", or "t2i-14B"
    "prompt": string,        // Required: Text description of the desired output
    "size": string,         // Optional: Output dimensions, default "1280*720"
    "frame_num": integer,   // Optional: Number of frames (default: 81 for video, 1 for image)
    "sample_steps": integer, // Optional: Number of sampling steps
    "sample_shift": float,  // Optional: Sampling shift factor
    "sample_solver": string, // Optional: "unipc" or "dpm++", default "unipc"
    "sample_guide_scale": float, // Optional: Classifier free guidance scale, default 5.0
    "base_seed": integer    // Optional: Random seed, default -1 (random)
}
```

#### Response

Returns the generated video (MP4) or image (PNG) file directly.

### Image-to-Video Generation

POST /generate/image

Generates a video from an input image and text prompt.

#### Request Body

Multipart form data with:
- `image`: File - The input image file
- `config`: JSON object containing:

```json
{
    "task": string,          // Required: Must be "i2v-14B"
    "prompt": string,        // Required: Text description of the desired output
    "size": string,         // Optional: Output dimensions, default "1280*720"
    "frame_num": integer,   // Optional: Number of frames, default 81
    "sample_steps": integer, // Optional: Number of sampling steps, default 40
    "sample_shift": float,  // Optional: Sampling shift factor
    "sample_solver": string, // Optional: "unipc" or "dpm++", default "unipc"
    "sample_guide_scale": float, // Optional: Classifier free guidance scale, default 5.0
    "base_seed": integer    // Optional: Random seed, default -1 (random)
}
```

#### Response

Returns the generated video file (MP4) directly.

## Supported Configurations

### Tasks
- `t2v-14B`: Text-to-video generation using 14B model
- `t2v-1.3B`: Text-to-video generation using 1.3B model
- `t2i-14B`: Text-to-image generation using 14B model
- `i2v-14B`: Image-to-video generation using 14B model

### Output Sizes
Different tasks support different output sizes. Check the API response for supported sizes for each task.

## Error Responses

### 400 Bad Request
Returned when:
- Invalid task specified
- Unsupported size for the task
- Invalid frame_num configuration
- Invalid input parameters

### 503 Service Unavailable
Returned when:
- GPU is not available for generation

### 500 Internal Server Error
Returned when:
- Generation process fails
- Unexpected server errors

## Example Usage

### Text-to-Video Generation

```bash
curl -X POST "http://localhost:6006/generate/text" \
     -H "Content-Type: application/json" \
     -d '{
           "task": "t2v-14B",
           "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage",
           "size": "1280*720"
         }'
```

### Image-to-Video Generation

```bash
curl -X POST "http://localhost:6006/generate/image" \
     -F "image=@/path/to/image.jpg" \
     -F 'config={"task": "i2v-14B", "prompt": "A cat surfing on the beach", "size": "1280*720"}'
```

## Requirements
- Environment variable `WAN_CKPT_DIR` must be set to the checkpoint directory path
- GPU with sufficient VRAM for generation tasks
- Output directory `outputs/` must be writable

