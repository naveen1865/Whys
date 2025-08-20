import { Module } from '@nestjs/common';
import { ReplyModule } from './reply/reply.module';

@Module({
  imports: [ReplyModule],
})
export class AppModule {}
