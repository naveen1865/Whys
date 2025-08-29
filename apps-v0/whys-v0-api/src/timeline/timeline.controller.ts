import { Controller, Get, Query } from '@nestjs/common';
import { db } from '../db';

@Controller('timeline')
export class TimelineController {
  @Get()
  async list(@Query('user_id') userId: string, @Query('limit') limit = '20') {
    const res = await db.query(
      `SELECT id, summary, action_items, tags, flags, created_at
       FROM memories WHERE user_id = $1
       ORDER BY created_at DESC
       LIMIT $2`,
      [userId, Number(limit)]
    );
    return { items: res.rows };
  }
}
