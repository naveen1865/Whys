// import { Controller } from '@nestjs/common';

// @Controller('ingest')
// export class IngestController {}


import { Body, Controller, Post } from '@nestjs/common';
import { z } from 'zod';
import { IngestService } from './ingest.service';

const EnqueueDto = z.object({ user_id: z.string(), s3_key: z.string() });

@Controller('ingest')
export class IngestController {
  constructor(private svc: IngestService) {}

  @Post('presign')
  async presign(@Body() body: { user_id: string }) {
    return this.svc.presignUpload(body.user_id);
  }

  @Post('enqueue')
  async enqueue(@Body() body: unknown) {
    const dto = EnqueueDto.parse(body);
    return this.svc.enqueueJob(dto.user_id, dto.s3_key);
  }
}
