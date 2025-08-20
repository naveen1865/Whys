import 'dotenv/config';

export default {
  expo: {
    name: "whys-v0-mobile",
    slug: "whys-v0-mobile",
    extra: {
      API_URL: process.env.API_URL || "http://localhost:5000"
    }
  }
};
