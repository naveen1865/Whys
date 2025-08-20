import * as Joi from 'joi';

export const envValidationSchema = Joi.object({
  PORT: Joi.number().default(5000),
  OPENAI_API_KEY: Joi.string().required(),
  TRANSCRIBE_MODEL: Joi.string().valid('whisper-1').default('whisper-1'),
  MAX_UPLOAD_MB: Joi.number().min(1).max(1000).default(25),
});
