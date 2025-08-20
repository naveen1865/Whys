import { Controller, Get } from '@nestjs/common';

@Controller()
export class AppController {
  @Get()
  health() {
    return { ok: true, service: 'whys-v0-api', ts: new Date().toISOString() };
  }
}
