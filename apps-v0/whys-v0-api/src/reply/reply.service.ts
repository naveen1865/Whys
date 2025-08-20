import { Injectable } from '@nestjs/common';

@Injectable()
export class ReplyService {
  reply(message: string): string {
    if (message?.toLowerCase() === 'hi') {
      return 'hello';
    }
    return 'bewakoof';
  }
}
