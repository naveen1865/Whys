import { Provider } from '@nestjs/common';
import OpenAI from 'openai';

export const OPENAI_CLIENT = Symbol('OPENAI_CLIENT');

export const OpenAIProvider = (apiKey: string): Provider => ({
  provide: OPENAI_CLIENT,
  useFactory: () => new OpenAI({ apiKey }),
});
