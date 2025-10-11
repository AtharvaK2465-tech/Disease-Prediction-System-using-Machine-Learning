const express = require('express');
const nodemailer = require('nodemailer');
const bodyParser = require('body-parser');

const app = express();

// Middleware
app.use(bodyParser.json());
app.use(express.static('public')); // serve your frontend files

// Contact form route
app.post('/send-message', async (req, res) => {
  const { name, email, message } = req.body;

  try {
    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: 'your.email@gmail.com',
        pass: 'your-app-password' // Use Gmail App Password
      }
    });

    // Admin email
    await transporter.sendMail({
      from: 'your.email@gmail.com',
      to: 'your.email@gmail.com',
      subject: `New contact from ${name}`,
      text: `Name: ${name}\nEmail: ${email}\nMessage: ${message}`
    });

    // User email
    await transporter.sendMail({
      from: 'your.email@gmail.com',
      to: email,
      replyTo: 'your.email@gmail.com',
      subject: `Thanks for contacting HealthPredictAI, ${name}!`,
      html: `
        <div style="font-family: sans-serif; color: #333;">
          <h2>Hi ${name},</h2>
          <p>Thank you for contacting <strong>HealthPredictAI</strong>!</p>
          <blockquote style="background: #f0f0f0; padding: 10px; border-radius: 5px;">
            ${message}
          </blockquote>
          <p>We will get back to you shortly.</p>
          <p>Best regards,<br/>HealthPredictAI Team</p>
        </div>
      `
    });

    res.json({ success: true, msg: 'Message sent! User also received a confirmation email.' });

  } catch (err) {
    console.error(err);
    res.json({ success: false, msg: 'Error sending emails.' });
  }
});

// Start server
app.listen(3000, () => console.log('Server running on http://localhost:3000'));
