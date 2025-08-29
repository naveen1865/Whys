import { Module } from '@nestjs/common';
import { TimelineController } from './timeline.controller';

@Module({
  controllers: [TimelineController],
})
export class TimelineModule {}
