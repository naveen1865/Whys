// import { Module } from '@nestjs/common';
// import { IngestController } from './ingest.controller';

// @Module({
//   controllers: [IngestController]
// })
// export class IngestModule {}

// src/ingest/ingest.module.ts
import { Module } from '@nestjs/common';
import { IngestController } from './ingest.controller';
import { IngestService } from './ingest.service';

@Module({
  controllers: [IngestController],
  providers: [IngestService],
})
export class IngestModule {}

