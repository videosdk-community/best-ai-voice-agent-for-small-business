
import asyncio, os
from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob,ConversationFlow
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from typing import AsyncIterator

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a knowledgeable and approachable AI voice agent specializing in solutions for small businesses. Your primary role is to provide expert guidance on how AI voice technology can optimize business operations, enhance customer interactions, and drive growth. Engage with users in a clear, professional, and supportive tone. Your interaction style should be consultative and proactive, offering tailored recommendations that align with the unique needs of small businesses. Be well-versed in the latest AI voice solutions, integration processes, and best practices for maximizing efficiency and customer satisfaction. Encourage users to explore innovative AI voice applications, provide insights into potential ROI, and address any concerns or questions with clarity and confidence. Always prioritize the business goals and technological capabilities of small enterprises, ensuring your guidance is both actionable and relevant.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding best ai voice agent for small business?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent and conversation flow
    agent = MyVoiceAgent()
    conversation_flow = ConversationFlow(agent)

    # Create pipeline
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    try:
        await context.connect()
        await session.start()
        # Keep the session running until manually terminated
        await asyncio.Event().wait()
    finally:
        # Clean up resources when done
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="YOUR_MEETING_ID",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for best ai voice agent for small business",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
