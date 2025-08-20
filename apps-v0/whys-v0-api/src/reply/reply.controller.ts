import { Controller, Post, Body } from '@nestjs/common';
import { ReplyService } from './reply.service';


@Controller('reply')
export class ReplyController {
  constructor(private readonly replyService: ReplyService) {}

  @Post()
  getReply(@Body('message') message: string) {
    return { reply: this.replyService.reply(message) };
  }
}
