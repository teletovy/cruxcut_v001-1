import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Facebook, Instagram, Twitter, Youtube } from "lucide-react"

export function Footer() {
  return (
    <footer className="bg-black border-t border-gray-800">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <Link href="/" className="flex items-center gap-2 mb-4">
              <span className="text-2xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
                CruxCut
              </span>
            </Link>
            <p className="text-gray-400 mb-6 max-w-md">
              CruxCut은 클라이머를 위한 AI 영상 편집 서비스입니다. 고정 카메라로 촬영한 영상을 역동적인 시점의 영상으로
              변환해 드립니다.
            </p>
            <div className="flex gap-4">
              <Button size="icon" variant="outline" className="rounded-full border-gray-800 hover:border-gray-700">
                <Facebook className="h-4 w-4" />
                <span className="sr-only">Facebook</span>
              </Button>
              <Button size="icon" variant="outline" className="rounded-full border-gray-800 hover:border-gray-700">
                <Instagram className="h-4 w-4" />
                <span className="sr-only">Instagram</span>
              </Button>
              <Button size="icon" variant="outline" className="rounded-full border-gray-800 hover:border-gray-700">
                <Twitter className="h-4 w-4" />
                <span className="sr-only">Twitter</span>
              </Button>
              <Button size="icon" variant="outline" className="rounded-full border-gray-800 hover:border-gray-700">
                <Youtube className="h-4 w-4" />
                <span className="sr-only">YouTube</span>
              </Button>
            </div>
          </div>
          <div>
            <h3 className="font-medium text-lg mb-4">서비스</h3>
            <ul className="space-y-3">
              <li>
                <Link href="#" className="text-gray-400 hover:text-white transition-colors">
                  기능
                </Link>
              </li>
              <li>
                <Link href="#" className="text-gray-400 hover:text-white transition-colors">
                  가격
                </Link>
              </li>
              <li>
                <Link href="#" className="text-gray-400 hover:text-white transition-colors">
                  FAQ
                </Link>
              </li>
              <li>
                <Link href="#" className="text-gray-400 hover:text-white transition-colors">
                  튜토리얼
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium text-lg mb-4">뉴스레터 구독</h3>
            <p className="text-gray-400 mb-4">최신 소식과 업데이트를 받아보세요.</p>
            <div className="flex gap-2">
              <Input placeholder="이메일 주소" className="bg-gray-900 border-gray-800" />
              <Button>구독</Button>
            </div>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">&copy; {new Date().getFullYear()} CruxCut. All rights reserved.</p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <Link href="#" className="text-gray-400 hover:text-white text-sm">
              개인정보처리방침
            </Link>
            <Link href="#" className="text-gray-400 hover:text-white text-sm">
              이용약관
            </Link>
            <Link href="#" className="text-gray-400 hover:text-white text-sm">
              문의하기
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}
