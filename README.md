# Inventory Shelf Management System

An AI-powered shelf management system that detects products on retail shelves, matches them against planogram layouts, and scores compliance in real time using computer vision models served via NVIDIA Triton, with GPU-accelerated preprocessing via NVIDIA DALI.

---

## Prerequisites

1. Docker & Docker Compose
2. NVIDIA GPU (recommended)
3. Valid `.env` — copy `.env.example` and fill in your values
4. Access to model repository and cloud storage bucket

---

## Setup

```bash
cp .env.example .env
# fill in .env

docker compose up --build -d
```

---

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/match_planogram` | — | Upload shelf images, receive Top-K planogram candidates |
| `POST` | `/confirm_planogram` | — | Confirm a candidate, receive final results and visualizations |
| `POST` | `/check_shelf` | — | Synchronous shelf compliance check |
| `POST` | `/check_shelf_async` | — | Async shelf compliance check |
| `GET` | `/task/{task_id}` | — | Poll async task status |
| `POST` | `/create_planogram_2d` | API key | Generate a 2D planogram image |
| `GET` | `/get_shelf_info` | API key | List shelf codes with signed URLs |
| `POST` | `/generate_label` | API key | Enqueue batch label translation |
| `GET` | `/generate_label_task/{task_id}` | API key | Poll label translation task |

Protected endpoints require `x-api-key: <API_KEY>` header.

---

## License

Proprietary Software License
Copyright (c) 2025 President Information Corp
All rights reserved.

This software and its associated documentation (the "Software") are the
confidential and proprietary information of President Information Corp ("PIC").
Except as expressly authorized in writing by PIC, you shall not, directly or
indirectly: (a) use, copy, modify, translate, or create derivative works of the
Software; (b) distribute, sublicense, sell, lease, lend, publish, disclose,
or otherwise make the Software available to any third party; (c) reverse
engineer, decompile, or disassemble the Software; or (d) remove or alter any
proprietary notices.

No license or other rights, express or implied, are granted by PIC under any
patents, copyrights, trade secrets, trademarks, or other intellectual property
rights, except as expressly set forth in a separate written agreement signed
by PIC.

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. TO THE MAXIMUM
EXTENT PERMITTED BY LAW, PIC DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED,
INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
AND NON-INFRINGEMENT. IN NO EVENT SHALL PIC BE LIABLE FOR ANY DAMAGES ARISING
OUT OF OR RELATED TO THE SOFTWARE OR ITS USE.

Unauthorized use or disclosure of the Software may cause irreparable harm to
PIC. PIC shall be entitled to seek injunctive relief, in addition to any other
remedies available at law or in equity.
